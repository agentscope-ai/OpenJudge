# -*- coding: utf-8 -*-
"""End-to-end pipeline for Reference Hallucination Arena.

Pipeline steps:
  1. Load queries from user-provided dataset
  2. Collect responses from target endpoints
  3. Extract BibTeX references from responses
  4. Verify references via Crossref / PubMed / arXiv / DBLP
  5. Compute objective scores and rankings
  6. Generate report and charts

Supports fine-grained, per-item checkpoint-based resume for all stages.
"""

import json
import random
import threading
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

from loguru import logger
from pydantic import BaseModel, Field

from cookbooks.ref_hallucination_arena.collectors.bib_extractor import BibExtractor
from cookbooks.ref_hallucination_arena.collectors.response_collector import (
    ResponseCollector,
)
from cookbooks.ref_hallucination_arena.loaders.dataset_loader import DatasetLoader
from cookbooks.ref_hallucination_arena.reporting.chart_generator import (
    RefChartGenerator,
)
from cookbooks.ref_hallucination_arena.reporting.report_generator import (
    RefReportGenerator,
)
from cookbooks.ref_hallucination_arena.schema import (
    ArenaResult,
    ModelVerificationResult,
    QueryItem,
    RefArenaConfig,
    VerificationStatus,
    load_config,
)
from cookbooks.ref_hallucination_arena.scoring.objective_scorer import ObjectiveScorer
from cookbooks.ref_hallucination_arena.scoring.ranking import RankingCalculator
from cookbooks.ref_hallucination_arena.verifiers.composite_verifier import (
    CompositeVerifier,
)

# =============================================================================
# Checkpoint Management
# =============================================================================


class PipelineStage(str, Enum):
    """Pipeline stages for checkpoint resume."""

    NOT_STARTED = "not_started"
    QUERIES_LOADED = "queries_loaded"
    RESPONSES_COLLECTING = "responses_collecting"  # in-progress (per-item)
    RESPONSES_COLLECTED = "responses_collected"
    REFS_EXTRACTED = "refs_extracted"
    VERIFICATION_IN_PROGRESS = "verification_in_progress"  # in-progress (per-item)
    VERIFICATION_COMPLETE = "verification_complete"
    EVALUATION_COMPLETE = "evaluation_complete"

    @classmethod
    def order(cls, stage: "PipelineStage") -> int:
        return {
            cls.NOT_STARTED: 0,
            cls.QUERIES_LOADED: 1,
            cls.RESPONSES_COLLECTING: 2,
            cls.RESPONSES_COLLECTED: 3,
            cls.REFS_EXTRACTED: 4,
            cls.VERIFICATION_IN_PROGRESS: 5,
            cls.VERIFICATION_COMPLETE: 6,
            cls.EVALUATION_COMPLETE: 7,
        }.get(stage, -1)

    def __ge__(self, other):
        return self.order(self) >= self.order(other)

    def __gt__(self, other):
        return self.order(self) > self.order(other)


class CheckpointData(BaseModel):
    """Checkpoint state with per-item progress tracking."""

    stage: PipelineStage = PipelineStage.NOT_STARTED
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    total_queries: int = 0

    # Per-item progress for Step 2: set of query indices whose responses are complete
    completed_response_indices: List[int] = Field(default_factory=list)

    # Per-item progress for Step 4: list of "model::query_idx" keys already verified
    completed_verification_keys: List[str] = Field(default_factory=list)


class CheckpointManager:
    """Manage pipeline checkpoints for resume capability.

    Supports both stage-level and per-item incremental checkpointing.
    """

    CHECKPOINT_FILE = "checkpoint.json"
    QUERIES_FILE = "queries.json"
    RESPONSES_FILE = "responses.json"
    EXTRACTED_FILE = "extracted_refs.json"
    VERIFICATION_FILE = "verification_results.json"

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._checkpoint: Optional[CheckpointData] = None
        self._lock = threading.Lock()

    def load(self) -> Optional[CheckpointData]:
        path = self.output_dir / self.CHECKPOINT_FILE
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                self._checkpoint = CheckpointData(**json.load(f))
            logger.info(f"Loaded checkpoint: stage={self._checkpoint.stage.value}")
            return self._checkpoint
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")
            return None

    def save(self, checkpoint: CheckpointData) -> None:
        checkpoint.updated_at = datetime.now().isoformat()
        self._checkpoint = checkpoint
        with self._lock:
            with open(self.output_dir / self.CHECKPOINT_FILE, "w", encoding="utf-8") as f:
                json.dump(checkpoint.model_dump(), f, indent=2, ensure_ascii=False)

    def update_stage(self, stage: PipelineStage, **kwargs) -> None:
        if self._checkpoint is None:
            self._checkpoint = CheckpointData()
        self._checkpoint.stage = stage
        for k, v in kwargs.items():
            if hasattr(self._checkpoint, k):
                setattr(self._checkpoint, k, v)
        self.save(self._checkpoint)

    def save_json(self, filename: str, data: Any) -> str:
        path = self.output_dir / filename
        with self._lock:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        return str(path)

    def load_json(self, filename: str) -> Any:
        path = self.output_dir / filename
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    # ---- Per-item incremental save helpers ----

    def mark_response_complete(self, query_idx: int) -> None:
        """Mark a single query's response collection as complete."""
        if self._checkpoint is None:
            self._checkpoint = CheckpointData()
        if query_idx not in self._checkpoint.completed_response_indices:
            self._checkpoint.completed_response_indices.append(query_idx)
        self.save(self._checkpoint)

    def get_completed_response_indices(self) -> Set[int]:
        """Return set of query indices whose responses have been collected."""
        if self._checkpoint is None:
            return set()
        return set(self._checkpoint.completed_response_indices)

    def mark_verification_complete_item(self, model_name: str, query_idx: int) -> None:
        """Mark a single (model, query_idx) verification as complete."""
        if self._checkpoint is None:
            self._checkpoint = CheckpointData()
        key = f"{model_name}::{query_idx}"
        if key not in self._checkpoint.completed_verification_keys:
            self._checkpoint.completed_verification_keys.append(key)
        self.save(self._checkpoint)

    def get_completed_verification_keys(self) -> Set[str]:
        """Return set of 'model::query_idx' keys already verified."""
        if self._checkpoint is None:
            return set()
        return set(self._checkpoint.completed_verification_keys)

    def save_responses_incremental(self, responses: List[Dict[str, Any]]) -> str:
        """Atomically save the full responses list (called after each query completes)."""
        return self.save_json(self.RESPONSES_FILE, responses)

    def save_verification_incremental(self, results: Dict[str, List[ModelVerificationResult]]) -> str:
        """Atomically save all verification results (called after each item completes)."""
        serialized = {model: [mvr.model_dump() for mvr in mvr_list] for model, mvr_list in results.items()}
        return self.save_json(self.VERIFICATION_FILE, serialized)

    def clear(self) -> None:
        for fn in [
            self.CHECKPOINT_FILE,
            self.QUERIES_FILE,
            self.RESPONSES_FILE,
            self.EXTRACTED_FILE,
            self.VERIFICATION_FILE,
        ]:
            p = self.output_dir / fn
            if p.exists():
                p.unlink()
        self._checkpoint = None
        logger.info("Checkpoint cleared")


# =============================================================================
# Main Pipeline
# =============================================================================


class RefArenaPipeline:
    """End-to-end Reference Hallucination Arena pipeline.

    Steps:
      1. Load queries from dataset
      2. Collect model responses  (per-query incremental checkpoint)
      3. Extract BibTeX references
      4. Verify references         (per-item incremental checkpoint)
      5. Score and rank models
      6. Generate report + charts

    All long-running stages (Step 2, Step 4) support fine-grained, per-item
    checkpoint resume so that interrupted runs lose at most one item of progress.

    Example:
        >>> pipeline = RefArenaPipeline.from_config("config.yaml")
        >>> result = await pipeline.evaluate()
        >>> print(result.rankings)
    """

    def __init__(self, config: RefArenaConfig, resume: bool = True):
        self.config = config
        self._resume = resume
        self._ckpt = CheckpointManager(config.output.output_dir)

        # Data holders
        self._queries: List[QueryItem] = []
        self._responses: List[Dict[str, Any]] = []
        self._extracted: Dict[str, Dict[str, List[dict]]] = {}  # {model: {query_idx: [ref_dicts]}}
        self._verification_results: Dict[str, List[ModelVerificationResult]] = {}

    @classmethod
    def from_config(cls, config_path: Union[str, Path], resume: bool = True) -> "RefArenaPipeline":
        config = load_config(config_path)
        return cls(config=config, resume=resume)

    # ---- Step 1: Load queries ----

    def _load_queries(self) -> List[QueryItem]:
        logger.info("Step 1: Loading queries from dataset...")
        loader = DatasetLoader(self.config.dataset.path)
        queries = loader.load()

        if self.config.dataset.shuffle:
            random.shuffle(queries)
        if self.config.dataset.max_queries:
            queries = queries[: self.config.dataset.max_queries]

        self._queries = queries
        logger.info(f"Loaded {len(queries)} queries")
        return queries

    # ---- Step 2: Collect responses (incremental) ----

    async def _collect_responses_incremental(
        self,
        completed_indices: Set[int],
    ) -> List[Dict[str, Any]]:
        """Collect model responses with per-query incremental checkpoint.

        Already-completed queries (from a previous interrupted run) are skipped.
        Each query's result is persisted to disk as soon as all its endpoints
        respond, so that a kill/crash loses at most one query's work.

        Args:
            completed_indices: Set of query indices already collected.

        Returns:
            Full list of response dicts (length == len(self._queries)).
        """
        logger.info("Step 2: Collecting model responses (incremental)...")
        total_queries = len(self._queries)
        pending_indices = [i for i in range(total_queries) if i not in completed_indices]

        if not pending_indices:
            logger.info("All responses already collected from checkpoint")
            return self._responses

        logger.info(
            f"  Total queries: {total_queries}, "
            f"already completed: {len(completed_indices)}, "
            f"remaining: {len(pending_indices)}"
        )

        collector = ResponseCollector(
            target_endpoints=self.config.target_endpoints,
            evaluation_config=self.config.evaluation,
        )

        # Build local→global index mapping
        local_to_global = {local_idx: global_idx for local_idx, global_idx in enumerate(pending_indices)}
        saved_count = len(completed_indices)

        def _on_query_complete(local_idx: int, result_dict: dict) -> None:
            """Callback fired as soon as one query's all endpoints are done."""
            nonlocal saved_count
            global_idx = local_to_global[local_idx]
            self._responses[global_idx] = result_dict
            self._ckpt.mark_response_complete(global_idx)
            self._ckpt.save_responses_incremental(self._responses)
            saved_count += 1
            if saved_count % 10 == 0 or saved_count == total_queries:
                logger.info(f"  Response progress: {saved_count}/{total_queries} queries saved")

        # Collect only pending queries, with per-query callback
        pending_queries = [self._queries[i] for i in pending_indices]
        pending_responses = await collector.collect(pending_queries, on_query_complete=_on_query_complete)

        # Final merge for any that might not have been saved via callback
        for local_idx, global_idx in enumerate(pending_indices):
            if not self._responses[global_idx]:
                self._responses[global_idx] = pending_responses[local_idx]

        return self._responses

    # ---- Step 3: Extract BibTeX ----

    def _extract_refs(self) -> Dict[str, Dict[str, List[dict]]]:
        """Extract BibTeX references from all responses.

        Returns:
            {model_name: {query_index: [reference_dicts]}}
        """
        logger.info("Step 3: Extracting BibTeX references from responses...")
        extractor = BibExtractor()
        extracted: Dict[str, Dict[str, List[dict]]] = {}

        for model_name in self.config.target_endpoints:
            extracted[model_name] = {}

        total_refs = 0
        for idx, resp_data in enumerate(self._responses):
            for model_name in self.config.target_endpoints:
                text = resp_data.get("responses", {}).get(model_name)
                if not text:
                    extracted[model_name][str(idx)] = []
                    continue
                refs = extractor.extract(text)
                extracted[model_name][str(idx)] = [r.model_dump() for r in refs]
                total_refs += len(refs)

        logger.info(f"Extracted {total_refs} references total across all models")
        self._extracted = extracted
        return extracted

    # ---- Step 4: Verify references (incremental) ----

    def _verify_single_query(
        self,
        verifier: CompositeVerifier,
        model_name: str,
        idx: int,
        resp_data: Dict[str, Any],
    ) -> ModelVerificationResult:
        """Verify references for a single (model, query) pair.

        Returns:
            ModelVerificationResult for this item.
        """
        from cookbooks.ref_hallucination_arena.schema import Reference

        query_text = resp_data["query"]
        discipline = resp_data.get("discipline")
        ref_dicts = self._extracted.get(model_name, {}).get(str(idx), [])

        # Retrieve year_constraint from the original query
        year_constraint = None
        if idx < len(self._queries):
            year_constraint = self._queries[idx].year_constraint

        refs = [Reference(**rd) for rd in ref_dicts]
        vr_list = verifier.verify_batch(refs, discipline=discipline) if refs else []

        # Compute per-query stats
        verified = sum(1 for v in vr_list if v.status == VerificationStatus.VERIFIED)
        suspect = sum(1 for v in vr_list if v.status == VerificationStatus.SUSPECT)
        not_found = sum(1 for v in vr_list if v.status == VerificationStatus.NOT_FOUND)
        errors = sum(1 for v in vr_list if v.status == VerificationStatus.ERROR)
        total = len(vr_list)
        confidences = [v.confidence for v in vr_list if v.confidence > 0]

        # Completeness
        comp_sum = 0.0
        for ref in refs:
            comp_sum += sum([bool(ref.doi), bool(ref.year), bool(ref.authors)]) / 3.0
        completeness = comp_sum / total if total > 0 else 0.0

        # Year constraint compliance
        has_yc = year_constraint is not None and year_constraint.is_set
        yc_compliant = 0
        yc_noncompliant = 0
        yc_unknown = 0
        yc_desc = ""

        if has_yc:
            yc_desc = year_constraint.describe()
            for vr in vr_list:
                matched_year = None
                if vr.match_detail and vr.match_detail.matched_year:
                    matched_year = vr.match_detail.matched_year
                elif vr.reference.year:
                    matched_year = vr.reference.year

                if matched_year is None:
                    yc_unknown += 1
                elif year_constraint.check(matched_year):
                    yc_compliant += 1
                else:
                    yc_noncompliant += 1

        mvr = ModelVerificationResult(
            model_name=model_name,
            query=query_text,
            discipline=discipline,
            total_refs=total,
            verified=verified,
            suspect=suspect,
            not_found=not_found,
            errors=errors,
            verification_rate=verified / total if total > 0 else 0.0,
            hallucination_rate=(suspect + not_found) / total if total > 0 else 0.0,
            avg_confidence=sum(confidences) / len(confidences) if confidences else 0.0,
            completeness=completeness,
            results=[v.model_dump() for v in vr_list],  # type: ignore
            has_year_constraint=has_yc,
            year_constraint_desc=yc_desc,
            year_compliant=yc_compliant,
            year_noncompliant=yc_noncompliant,
            year_unknown=yc_unknown,
            year_compliance_rate=(yc_compliant / total if has_yc and total > 0 else 0.0),
        )

        yc_info = f" year_ok={yc_compliant}/{total}" if has_yc else ""
        logger.debug(f"  {model_name} | Q{idx}: {verified}/{total} verified" f"{yc_info} ({discipline or 'unknown'})")
        return mvr

    def _verify_refs_incremental(
        self,
        completed_keys: Set[str],
    ) -> Dict[str, List[ModelVerificationResult]]:
        """Verify references with per-(model, query) incremental checkpoint.

        Uses a thread pool to verify multiple queries in parallel, with a
        semaphore to cap total concurrent API calls to external services
        (Crossref, PubMed, arXiv, DBLP).

        Already-verified items (from a previous interrupted run) are skipped.
        After each item is verified, results are persisted so a subsequent
        resume will not repeat it.

        Args:
            completed_keys: Set of "model::query_idx" keys already verified.

        Returns:
            {model_name: [ModelVerificationResult, ...]}
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        logger.info("Step 4: Verifying references (parallel, incremental)...")

        total_items = len(self.config.target_endpoints) * len(self._responses)
        already_done = len(completed_keys)
        remaining = total_items - already_done
        logger.info(f"  Total items: {total_items}, " f"already verified: {already_done}, " f"remaining: {remaining}")

        results = self._verification_results
        # Lock for thread-safe checkpoint writes and results updates
        ckpt_lock = threading.Lock()
        done_count = already_done

        # Determine parallel workers: use verification max_workers (default 10)
        # This controls how many queries are verified concurrently at the outer
        # level. Each query's internal verify_batch uses its own concurrency
        # but shares the same HTTP clients, so total API pressure is bounded.
        parallel_queries = self.config.verification.max_workers or 10

        # Build list of (model_name, idx) work items that need verification
        work_items = []
        for model_name in self.config.target_endpoints:
            if model_name not in results:
                results[model_name] = [None] * len(self._responses)  # type: ignore
            for idx in range(len(self._responses)):
                key = f"{model_name}::{idx}"
                if key not in completed_keys:
                    work_items.append((model_name, idx))

        logger.info(
            f"  Parallel verification: {len(work_items)} items to verify " f"with {parallel_queries} concurrent workers"
        )

        with CompositeVerifier(config=self.config.verification) as verifier:

            def _verify_one(model_name: str, idx: int) -> None:
                nonlocal done_count
                resp_data = self._responses[idx]
                mvr = self._verify_single_query(verifier, model_name, idx, resp_data)

                with ckpt_lock:
                    results[model_name][idx] = mvr
                    self._ckpt.mark_verification_complete_item(model_name, idx)
                    safe_results: Dict[str, List[ModelVerificationResult]] = {
                        m: [r for r in rlist if r is not None] for m, rlist in results.items()
                    }
                    self._ckpt.save_verification_incremental(safe_results)
                    done_count += 1
                    if done_count % 50 == 0 or done_count == total_items:
                        logger.info(f"  Verification progress: {done_count}/{total_items} items saved")

            with ThreadPoolExecutor(max_workers=parallel_queries) as executor:
                futures = {executor.submit(_verify_one, mn, idx): (mn, idx) for mn, idx in work_items}
                for future in as_completed(futures):
                    mn, idx = futures[future]
                    try:
                        future.result()
                    except Exception as e:
                        logger.error(f"  Verification failed for {mn} Q{idx}: {e}")

        # Log per-model summary
        for model_name in self.config.target_endpoints:
            model_results = [r for r in results.get(model_name, []) if r is not None]
            total_v = sum(m.verified for m in model_results)
            total_r = sum(m.total_refs for m in model_results)
            if total_r > 0:
                logger.info(f"  {model_name}: {total_v}/{total_r} verified " f"({total_v/total_r:.1%})")
            else:
                logger.info(f"  {model_name}: 0 refs")

        # Clean up None placeholders (should not remain, but be safe)
        self._verification_results = {m: [r for r in rlist if r is not None] for m, rlist in results.items()}
        return self._verification_results

    # ---- Step 5: Score and rank ----

    def _score_and_rank(self) -> ArenaResult:
        logger.info("Step 5: Computing scores and rankings...")
        scorer = ObjectiveScorer()
        model_scores = scorer.score_all_models(self._verification_results)

        ranker = RankingCalculator()
        result = ranker.calculate(model_scores)
        result.total_queries = len(self._queries)
        return result

    # ---- Step 6: Report + Charts ----

    def _generate_report(self, result: ArenaResult) -> Optional[str]:
        if not self.config.report.enabled:
            return None

        logger.info("Step 6: Generating report...")
        generator = RefReportGenerator(
            language=self.config.report.language,
            include_examples=self.config.report.include_examples,
        )
        report = generator.generate(
            config=self.config,
            result=result,
            all_verification_details=self._verification_results,
        )

        report_path = Path(self.config.output.output_dir) / "evaluation_report.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
        logger.info(f"Report saved to {report_path}")
        return str(report_path)

    def _generate_charts(self, result: ArenaResult) -> None:
        if not self.config.report.enabled or not self.config.report.chart.enabled:
            return

        logger.info("Generating charts...")
        generator = RefChartGenerator(config=self.config.report.chart)
        generator.generate_verification_chart(result, self.config.output.output_dir)
        generator.generate_discipline_chart(result, self.config.output.output_dir)

    # ---- Main evaluate ----

    async def evaluate(self) -> ArenaResult:
        """Run the full evaluation pipeline with fine-grained checkpoint support.

        Each long-running stage (response collection, verification) saves progress
        per item, so interrupted runs can resume from the exact point of interruption.

        Returns:
            ArenaResult with rankings and scores.
        """
        checkpoint = self._ckpt.load() if self._resume else None

        # Step 1: Load queries
        if checkpoint and checkpoint.stage >= PipelineStage.QUERIES_LOADED:
            data = self._ckpt.load_json(CheckpointManager.QUERIES_FILE)
            self._queries = [QueryItem(**q) for q in data] if data else []
            logger.info(f"Resumed {len(self._queries)} queries from checkpoint")
        else:
            self._load_queries()
            self._ckpt.save_json(
                CheckpointManager.QUERIES_FILE,
                [q.model_dump() for q in self._queries],
            )
            self._ckpt.update_stage(
                PipelineStage.QUERIES_LOADED,
                total_queries=len(self._queries),
            )

        # Step 2: Collect responses (incremental)
        if checkpoint and checkpoint.stage >= PipelineStage.RESPONSES_COLLECTED:
            self._responses = self._ckpt.load_json(CheckpointManager.RESPONSES_FILE) or []
            logger.info(f"Resumed {len(self._responses)} responses from checkpoint (complete)")
        else:
            # Load partial responses if stage is RESPONSES_COLLECTING
            completed_indices: Set[int] = set()
            if checkpoint and checkpoint.stage >= PipelineStage.RESPONSES_COLLECTING:
                partial = self._ckpt.load_json(CheckpointManager.RESPONSES_FILE)
                if partial:
                    self._responses = partial
                    completed_indices = self._ckpt.get_completed_response_indices()
                    logger.info(
                        f"Resuming response collection: " f"{len(completed_indices)}/{len(self._queries)} already done"
                    )
                else:
                    self._responses = [{}] * len(self._queries)
            else:
                # Fresh start: initialize empty response slots
                self._responses = [{}] * len(self._queries)

            self._ckpt.update_stage(PipelineStage.RESPONSES_COLLECTING)
            await self._collect_responses_incremental(completed_indices)
            self._ckpt.save_json(CheckpointManager.RESPONSES_FILE, self._responses)
            self._ckpt.update_stage(PipelineStage.RESPONSES_COLLECTED)

        # Step 3: Extract refs
        if checkpoint and checkpoint.stage >= PipelineStage.REFS_EXTRACTED:
            self._extracted = self._ckpt.load_json(CheckpointManager.EXTRACTED_FILE) or {}
            total = sum(len(refs) for model_refs in self._extracted.values() for refs in model_refs.values())
            logger.info(f"Resumed {total} extracted refs from checkpoint")
        else:
            self._extract_refs()
            self._ckpt.save_json(CheckpointManager.EXTRACTED_FILE, self._extracted)
            self._ckpt.update_stage(PipelineStage.REFS_EXTRACTED)

        # Step 4: Verify refs (incremental)
        if checkpoint and checkpoint.stage >= PipelineStage.VERIFICATION_COMPLETE:
            raw = self._ckpt.load_json(CheckpointManager.VERIFICATION_FILE) or {}
            self._verification_results = {
                model: [ModelVerificationResult(**mvr) for mvr in mvr_list] for model, mvr_list in raw.items()
            }
            logger.info("Resumed verification results from checkpoint (complete)")
        else:
            # Load partial verification results if stage is VERIFICATION_IN_PROGRESS
            completed_v_keys: Set[str] = set()
            if checkpoint and checkpoint.stage >= PipelineStage.VERIFICATION_IN_PROGRESS:
                raw = self._ckpt.load_json(CheckpointManager.VERIFICATION_FILE) or {}
                self._verification_results = {
                    model: [ModelVerificationResult(**mvr) for mvr in mvr_list] for model, mvr_list in raw.items()
                }
                completed_v_keys = self._ckpt.get_completed_verification_keys()
                logger.info(f"Resuming verification: " f"{len(completed_v_keys)} items already verified")
            else:
                self._verification_results = {}

            self._ckpt.update_stage(PipelineStage.VERIFICATION_IN_PROGRESS)
            self._verify_refs_incremental(completed_v_keys)
            serialized = {
                model: [mvr.model_dump() for mvr in mvr_list] for model, mvr_list in self._verification_results.items()
            }
            self._ckpt.save_json(CheckpointManager.VERIFICATION_FILE, serialized)
            self._ckpt.update_stage(PipelineStage.VERIFICATION_COMPLETE)

        # Step 5: Score and rank
        result = self._score_and_rank()

        # Save final results
        self._ckpt.save_json(
            "evaluation_results.json",
            result.model_dump(),
        )
        self._ckpt.update_stage(PipelineStage.EVALUATION_COMPLETE)

        # Step 6: Report + Charts
        self._generate_report(result)
        self._generate_charts(result)

        return result

    def save_results(self, result: ArenaResult) -> Path:
        """Save results to output directory."""
        output_dir = Path(self.config.output.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / "evaluation_results.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result.model_dump(), f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to {output_file}")
        return output_file

    def clear_checkpoint(self) -> None:
        """Clear all checkpoint data."""
        self._ckpt.clear()
