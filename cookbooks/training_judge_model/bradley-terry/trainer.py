#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FSDP Bradley-Terry Trainer for Judge Models
Based on the VERL FSDP SFT Trainer architecture
"""

import logging
import os
from typing import Any, Dict, List

import hydra
import torch
import torch.distributed
from tensordict import TensorDict
from torch import optim
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import CPUOffload
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForSequenceClassification
from verl.utils.debug import log_gpu_memory_usage
from verl.utils.device import get_device_name, get_torch_device, is_cuda_available
from verl.utils.distributed import (
    destroy_global_process_group,
    initialize_global_process_group,
)
from verl.utils.fs import copy_to_local
from verl.utils.fsdp_utils import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    apply_fsdp2,
    fsdp2_clip_grad_norm_,
    fsdp2_load_full_state_dict,
    get_fsdp_wrap_policy,
    get_init_weight_context_manager,
    init_fn,
)
from verl.utils.import_utils import load_extern_type
from verl.utils.torch_dtypes import PrecisionType
from verl.utils.torch_functional import (
    get_cosine_schedule_with_warmup,
    get_wsd_schedule_with_warmup,
)
from verl.utils.tracking import Tracking

os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "true"


logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_BT_LOGGING_LEVEL", "WARN"))


def bt_collate_fn(features: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function for Bradley-Terry training
    Merges chosen and rejected responses into a single batch
    Since dataset now returns fixed-length tensors, we just need to stack them
    """
    # Merge chosen and rejected responses
    input_ids_list = []
    attention_mask_list = []

    for feature in features:
        # Add chosen response
        input_ids_list.append(feature["input_ids_j"])
        attention_mask_list.append(feature["attention_mask_j"])
        # Add rejected response
        input_ids_list.append(feature["input_ids_k"])
        attention_mask_list.append(feature["attention_mask_k"])

    return {
        "input_ids": torch.stack(input_ids_list),
        "attention_mask": torch.stack(attention_mask_list),
    }


class FSDPBTTrainer:
    """FSDP Bradley-Terry Trainer"""

    def __init__(
        self,
        config,
        device_mesh: DeviceMesh,
        tokenizer,
        train_dataset: Dataset,
        val_dataset: Dataset,
    ):
        self.config = config
        self.device_mesh = device_mesh
        self.tokenizer = tokenizer

        # Normalize dp size
        self._normalize_config_bsz()

        self._build_dataloader(train_dataset, val_dataset)
        self._build_model_optimizer()

        if self.device_mesh.get_rank() == 0:
            print(self.config)
        # Use device from config if available, otherwise auto-detect
        self.device_name = getattr(self.config.trainer, "device", get_device_name())

    def _normalize_config_bsz(self):
        dp_size = self.device_mesh.size(0)
        if self.device_mesh.get_rank() == 0:
            print(f"Normalize batch size by dp {dp_size}")

        assert (
            self.config.data.train_batch_size % dp_size == 0
        ), f"Global batch size {self.config.data.train_batch_size} is not divisible by dp size {dp_size}"

        self.config.data.train_batch_size //= dp_size

        assert self.config.data.train_batch_size % self.config.data.micro_batch_size_per_gpu == 0

    def _build_dataloader(self, train_dataset, val_dataset):
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        rank = self.device_mesh.get_rank()
        world_size = self.device_mesh.size()

        self.train_sampler = DistributedSampler(
            self.train_dataset,
            shuffle=True,
            num_replicas=world_size,
            rank=rank,
            drop_last=True,
        )
        self.train_dataloader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.train_batch_size,
            sampler=self.train_sampler,
            collate_fn=bt_collate_fn,
            num_workers=8,
            pin_memory=True,
            drop_last=True,
        )

        self.val_sampler = DistributedSampler(
            self.val_dataset,
            shuffle=False,
            num_replicas=world_size,
            rank=rank,
            drop_last=True,
        )
        self.val_dataloader = DataLoader(
            dataset=self.val_dataset,
            batch_size=self.config.data.micro_batch_size_per_gpu,
            sampler=self.val_sampler,
            collate_fn=bt_collate_fn,
            num_workers=8,
            pin_memory=True,
            drop_last=True,
        )

    def _build_model_optimizer(self):
        local_model_path = copy_to_local(
            src=self.config.model.partial_pretrain,
            verbose=True,
        )

        if self.config.model.external_lib:
            import importlib

            importlib.import_module(self.config.model.external_lib)

        log_gpu_memory_usage("Before model allocation", logger=logger)

        trust_remote_code = self.config.model.trust_remote_code
        torch_dtype = self.config.model.fsdp_config.get("model_dtype", "bf16")
        torch_dtype = PrecisionType.to_dtype(torch_dtype)

        # Load config first
        config = AutoConfig.from_pretrained(
            local_model_path,
            trust_remote_code=trust_remote_code,
        )
        config.pad_token_id = self.tokenizer.pad_token_id
        config.num_labels = 1  # Single score output
        config.problem_type = "regression"  # Regression task for scoring

        # Set use_cache=False when gradient checkpointing is enabled to avoid warning
        if self.config.model.enable_gradient_checkpointing:
            config.use_cache = False

        self.model_config = config

        # Use init context manager
        init_context = get_init_weight_context_manager(
            use_meta_tensor=not config.tie_word_embeddings,
            mesh=self.device_mesh,
        )

        with init_context():
            # Create judge model using AutoModelForSequenceClassification
            self.model = AutoModelForSequenceClassification.from_pretrained(
                local_model_path,
                config=config,
                torch_dtype=torch_dtype,
                trust_remote_code=trust_remote_code,
                ignore_mismatched_sizes=True,  # Allow classifier head size mismatch
            )

            # Apply Liger kernel if enabled
            if self.config.model.use_liger:
                from liger_kernel.transformers.monkey_patch import (
                    _apply_liger_kernel_to_instance,
                )

                _apply_liger_kernel_to_instance(model=self.model)

        if self.config.model.enable_gradient_checkpointing:
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False},
            )

        log_gpu_memory_usage("After model allocation", logger=logger)

        # FSDP wrapping
        mixed_precision = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.float32,
        )

        auto_wrap_policy = get_fsdp_wrap_policy(
            self.model,
            config=self.config.model.fsdp_config.get("wrap_policy", {}),
            is_lora=self.config.model.lora_rank > 0,
        )

        if not self.config.model.fsdp_config.get("cpu_offload", False):
            cpu_offload = None
        else:
            cpu_offload = CPUOffload(
                offload_params=self.config.model.fsdp_config.get(
                    "offload_params",
                    True,
                ),
            )

        fsdp_strategy = self.config.model.strategy
        if fsdp_strategy == "fsdp":
            self.fsdp_model = FSDP(
                self.model,
                cpu_offload=cpu_offload,
                param_init_fn=init_fn,
                use_orig_params=False,
                auto_wrap_policy=auto_wrap_policy,
                device_id=get_torch_device().current_device(),
                sharding_strategy=ShardingStrategy.FULL_SHARD,
                mixed_precision=mixed_precision,
                sync_module_states=True,
                device_mesh=self.device_mesh,
                forward_prefetch=False,
            )
        elif fsdp_strategy == "fsdp2":
            assert CPUOffloadPolicy is not None, "PyTorch version >= 2.4 is required for FSDP2"
            mp_policy = MixedPrecisionPolicy(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32,
                cast_forward_inputs=True,
            )

            fsdp_kwargs = {
                "mesh": self.device_mesh,
                "mp_policy": mp_policy,
                "offload_policy": cpu_offload,
                "reshard_after_forward": True,
            }
            full_state = self.model.state_dict()
            apply_fsdp2(self.model, fsdp_kwargs, self.config.model.fsdp_config)
            fsdp2_load_full_state_dict(
                self.model,
                full_state,
                self.device_mesh,
                cpu_offload,
            )
            self.fsdp_model = self.model
        else:
            raise NotImplementedError(f"Strategy {fsdp_strategy} not implemented")

        log_gpu_memory_usage("After FSDP wrapping", logger=logger)

        # Optimizer - use standard AdamW
        self.optimizer = optim.AdamW(
            self.fsdp_model.parameters(),
            lr=self.config.optim.lr,
            betas=tuple(self.config.optim.betas),
            weight_decay=self.config.optim.weight_decay,
        )

        log_gpu_memory_usage("After initialize optimizer", logger=logger)

        # Learning rate scheduler
        self.steps_per_epoch = len(self.train_dataloader)
        self.total_steps = self.steps_per_epoch * self.config.trainer.total_epochs

        if self.device_mesh.get_rank() == 0:
            print(
                f"Steps/epoch: {self.steps_per_epoch}, epochs: {self.config.trainer.total_epochs}, total steps: {self.total_steps}",
            )

        # Get warmup ratio from config
        warmup_ratio = self.config.optim.get("warmup_steps_ratio", 0.03)
        num_warmup_steps = int(self.total_steps * warmup_ratio)

        # Get lr scheduler type from config
        lr_scheduler_type = self.config.optim.get("lr_scheduler", "cosine")

        if lr_scheduler_type == "cosine":
            self.lr_scheduler = get_cosine_schedule_with_warmup(
                optimizer=self.optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=self.total_steps,
            )
        elif lr_scheduler_type == "wsd":
            self.lr_scheduler = get_wsd_schedule_with_warmup(
                optimizer=self.optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=self.total_steps,
            )
        elif lr_scheduler_type == "constant":
            # Constant LR after warmup
            from torch.optim.lr_scheduler import LambdaLR

            def lr_lambda(current_step: int):
                if current_step < num_warmup_steps:
                    return float(current_step) / float(max(1, num_warmup_steps))
                return 1.0

            self.lr_scheduler = LambdaLR(self.optimizer, lr_lambda)
        else:
            raise ValueError(f"Unknown lr scheduler: {lr_scheduler_type}")

    def _compute_bradley_terry_loss(self, batch, do_backward=True):
        """Compute Bradley-Terry loss"""
        input_ids = batch["input_ids"].to(self.device_name)
        attention_mask = batch["attention_mask"].to(self.device_name)

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            outputs = self.fsdp_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            logits = outputs.logits.squeeze(
                -1,
            )  # Shape: (batch_size, 1) -> (batch_size,)

            # Split into chosen and rejected pairs
            # Since bt_collate_fn creates interleaved batch [chosen_1, rejected_1, chosen_2, rejected_2, ...]
            # we need to use strided slicing to correctly separate them
            batch_size = logits.size(0) // 2

            chosen_rewards = logits[0::2]  # Take every 2nd element starting from 0: chosen responses
            rejected_rewards = logits[1::2]  # Take every 2nd element starting from 1: rejected responses

            # Check if we have valid pairs
            if batch_size == 0:
                logger.error(
                    "ERROR: batch_size is 0, cannot compute Bradley-Terry loss!",
                )
                return torch.tensor(
                    0.0,
                    device=logits.device,
                    requires_grad=True,
                ), torch.tensor(0.0, device=logits.device)

            # Bradley-Terry loss: -log(sigmoid(r_chosen - r_rejected))
            loss = -torch.nn.functional.logsigmoid(
                chosen_rewards - rejected_rewards,
            ).mean()

            # Accuracy
            accuracy = (chosen_rewards > rejected_rewards).float().mean()

            if do_backward:
                loss.backward()

            return loss, accuracy

    def training_step(self, batch: TensorDict):
        self.fsdp_model.train()

        log_gpu_memory_usage("Before optimizer zero_grad", logger=logger)
        self.optimizer.zero_grad()
        log_gpu_memory_usage("After optimizer zero_grad", logger=logger)

        # For Bradley-Terry training, micro_batch_size should consider pairs (chosen + rejected)
        # Each preference pair produces 2 samples, so micro batch size should be even
        micro_batch_size = self.config.data.micro_batch_size_per_gpu * 2  # Account for pairs

        micro_batches = batch.split(micro_batch_size)
        n_micro_batches = len(micro_batches)

        step_loss = 0
        step_accuracy = 0

        for i, micro_batch in enumerate(micro_batches):
            loss, accuracy = self._compute_bradley_terry_loss(batch=micro_batch, do_backward=False)
            # Scale loss for gradient accumulation
            scaled_loss = loss / n_micro_batches
            scaled_loss.backward()
            # Accumulate unscaled loss for logging
            step_loss += loss.item()
            step_accuracy += accuracy.item()

        # Gradient clipping
        if self.config.model.strategy == "fsdp":
            grad_norm = self.fsdp_model.clip_grad_norm_(
                max_norm=self.config.optim.clip_grad,
            )
        elif self.config.model.strategy == "fsdp2":
            grad_norm = fsdp2_clip_grad_norm_(
                self.fsdp_model.parameters(),
                max_norm=self.config.optim.clip_grad,
            )
        else:
            raise NotImplementedError(
                f"Strategy {self.config.model.strategy} not implemented",
            )

        log_gpu_memory_usage("Before optimizer step", logger=logger)

        # Check gradient norm
        if not torch.isfinite(grad_norm):
            print(f"WARN: grad_norm is not finite: {grad_norm}")
            self.optimizer.zero_grad()
        else:
            self.optimizer.step()

        log_gpu_memory_usage("After optimizer step", logger=logger)

        self.lr_scheduler.step()
        lr = self.lr_scheduler.get_last_lr()[0]

        # Average metrics across micro batches
        step_loss = step_loss / n_micro_batches
        step_accuracy = step_accuracy / n_micro_batches

        # Reduce loss across ranks
        step_loss = torch.tensor(step_loss).to(self.device_name)
        step_accuracy = torch.tensor(step_accuracy).to(self.device_name)

        if is_cuda_available:
            torch.distributed.all_reduce(step_loss, op=torch.distributed.ReduceOp.AVG)
            torch.distributed.all_reduce(
                step_accuracy,
                op=torch.distributed.ReduceOp.AVG,
            )

        return {
            "train/loss": step_loss.detach().item(),
            "train/accuracy": step_accuracy.detach().item(),
            "train/lr(1e-3)": lr * 1e3,
        }

    def validation_step(self, batch: TensorDict):
        self.fsdp_model.eval()
        with torch.no_grad():
            loss, accuracy = self._compute_bradley_terry_loss(batch, do_backward=False)

            if is_cuda_available:
                torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.AVG)
                torch.distributed.all_reduce(
                    accuracy,
                    op=torch.distributed.ReduceOp.AVG,
                )

        return {"val/loss": loss.item(), "val/accuracy": accuracy.item()}

    def save_checkpoint(self, step):
        path = os.path.join(
            self.config.trainer.default_local_dir,
            f"global_step_{step}",
        )

        fsdp_strategy = self.config.model.strategy
        if fsdp_strategy == "fsdp":
            from torch.distributed.fsdp import FullStateDictConfig, StateDictType

            cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(
                self.fsdp_model,
                StateDictType.FULL_STATE_DICT,
                cfg,
            ):
                state_dict = self.fsdp_model.state_dict()

            if self.device_mesh.get_rank() == 0:
                os.makedirs(path, exist_ok=True)
                self.model.save_pretrained(path, state_dict=state_dict)
                self.tokenizer.save_pretrained(path)

        elif fsdp_strategy == "fsdp2":
            from torch.distributed.checkpoint.state_dict import (
                StateDictOptions,
                get_model_state_dict,
            )

            options = StateDictOptions(full_state_dict=True, cpu_offload=True)
            state_dict = get_model_state_dict(self.fsdp_model, options=options)

            if self.device_mesh.get_rank() == 0:
                os.makedirs(path, exist_ok=True)
                self.model.save_pretrained(path, state_dict=state_dict)
                self.model_config.save_pretrained(path)
                self.tokenizer.save_pretrained(path)

        # Copy to HDFS if configured
        if self.device_mesh.get_rank() == 0 and self.config.trainer.default_hdfs_dir:
            import verl.utils.hdfs_io as hdfs_io

            hdfs_io.makedirs(self.config.trainer.default_hdfs_dir, exist_ok=True)
            hdfs_io.copy(
                src=path,
                dst=self.config.trainer.default_hdfs_dir,
                dirs_exist_ok=True,
            )

        torch.distributed.barrier()

    def fit(self):
        rank = self.device_mesh.get_rank()

        if rank == 0:
            tracking = Tracking(
                project_name=self.config.trainer.project_name,
                experiment_name=self.config.trainer.experiment_name,
                default_backend=self.config.trainer.logger,
            )

        global_step = 0
        last_valid_metric = None
        latest_train_metric = {}

        if self.config.trainer.total_training_steps:
            self.total_training_steps = self.config.trainer.total_training_steps
        else:
            self.total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs
        print(f"Total training steps: {self.total_training_steps}")

        # Create a single progress bar for all training steps
        if rank == 0:
            pbar = tqdm(total=self.total_training_steps, desc="Training")

        for epoch in range(self.config.trainer.total_epochs):
            self.train_sampler.set_epoch(epoch=epoch)
            for data in self.train_dataloader:
                global_step += 1
                # Bradley-Terry training: each preference pair becomes 2 samples (chosen + rejected)
                actual_batch_size = self.config.data.train_batch_size * 2
                data = TensorDict(data, batch_size=actual_batch_size).to(
                    self.device_name,
                )
                metric = self.training_step(data)

                if rank == 0:
                    tracking.log(data=metric, step=global_step)
                    # Update progress bar with current metrics
                    pbar.set_postfix(
                        {
                            "loss": f"{metric['train/loss']:.3f}",
                            "acc": f"{metric['train/accuracy']:.3f}",
                            "lr": f"{metric['train/lr(1e-3)']:.3f}",
                            "epoch": f"{epoch + 1}/{self.config.trainer.total_epochs}",
                        },
                    )
                    pbar.update(1)

                # Store latest training metrics for validation display
                latest_train_metric = metric

                is_last_step = global_step >= self.total_training_steps
                is_valid_step = global_step % self.config.trainer.test_freq == 0
                is_save_step = global_step % self.config.trainer.save_freq == 0

                # Validation
                if is_last_step or (self.config.trainer.test_freq > 0 and is_valid_step):
                    val_metrics = []
                    for val_data in self.val_dataloader:
                        # Bradley-Terry validation: each preference pair becomes 2 samples
                        val_actual_batch_size = self.config.data.micro_batch_size_per_gpu * 2
                        val_data = TensorDict(
                            val_data,
                            batch_size=val_actual_batch_size,
                        ).to(self.device_name)
                        val_metric = self.validation_step(val_data)
                        val_metrics.append(val_metric)

                    if rank == 0:
                        avg_val_loss = sum(m["val/loss"] for m in val_metrics) / len(
                            val_metrics,
                        )
                        avg_val_accuracy = sum(m["val/accuracy"] for m in val_metrics) / len(val_metrics)

                        metric = {
                            "val/loss": avg_val_loss,
                            "val/accuracy": avg_val_accuracy,
                        }
                        tracking.log(data=metric, step=global_step)
                        last_valid_metric = metric
                        # Update progress bar with validation metrics
                        postfix_dict = {
                            "val_loss": f"{avg_val_loss:.3f}",
                            "val_acc": f"{avg_val_accuracy:.3f}",
                            "epoch": f"{epoch + 1}/{self.config.trainer.total_epochs}",
                        }
                        # Add training metrics if available
                        if latest_train_metric:
                            postfix_dict.update(
                                {
                                    "loss": f"{latest_train_metric['train/loss']:.3f}",
                                    "acc": f"{latest_train_metric['train/accuracy']:.3f}",
                                },
                            )
                        pbar.set_postfix(postfix_dict)

                    torch.distributed.barrier()

                # Save checkpoint
                if is_last_step or (self.config.trainer.save_freq > 0 and is_save_step):
                    self.save_checkpoint(step=global_step)

                if is_last_step:
                    if rank == 0:
                        pbar.close()
                        print(f"Final validation metrics: {last_valid_metric}")
                    return


def create_bt_dataset(train_files, val_files, data_config, tokenizer):
    """Create Bradley-Terry dataset with support for custom dataset classes"""

    # First check if a custom dataset class is specified
    if data_config.custom_cls.get("path", None):
        # Load custom dataset class from external module
        dataset_cls = load_extern_type(
            data_config.custom_cls.path,
            data_config.custom_cls.name,
        )
        print(
            f"Using custom dataset class: {data_config.custom_cls.name} from {data_config.custom_cls.path}",
        )
    else:
        # Default to built-in BTDataset
        from .dataset import BTDataset

        dataset_cls = BTDataset
        print("Using default BTDataset class")

    # Create datasets
    train_dataset = dataset_cls(
        parquet_files=train_files,
        tokenizer=tokenizer,
        config=data_config,
    )
    val_dataset = dataset_cls(
        parquet_files=val_files,
        tokenizer=tokenizer,
        config=data_config,
    )

    return train_dataset, val_dataset


def run_bt_training(config):
    """Main Bradley-Terry training function"""
    device_name = get_device_name()
    local_rank, rank, world_size = initialize_global_process_group()

    device_mesh = init_device_mesh(
        device_type=device_name,
        mesh_shape=(world_size,),
        mesh_dim_names=("fsdp",),
    )

    # Build tokenizer and datasets
    from verl.utils import hf_tokenizer

    local_model_path = copy_to_local(src=config.model.partial_pretrain, verbose=True)
    tokenizer = hf_tokenizer(
        local_model_path,
        trust_remote_code=config.model.trust_remote_code,
    )

    # Ensure pad token exists
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    # Create datasets
    train_dataset, val_dataset = create_bt_dataset(
        config.data.train_files,
        config.data.val_files,
        config.data,
        tokenizer,
    )

    # Create trainer
    trainer = FSDPBTTrainer(
        config=config,
        device_mesh=device_mesh,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )

    # Start training
    trainer.fit()

    destroy_global_process_group()


@hydra.main(config_path=".", config_name="trainer", version_base=None)
def main(config):
    run_bt_training(config)


if __name__ == "__main__":
    main()
