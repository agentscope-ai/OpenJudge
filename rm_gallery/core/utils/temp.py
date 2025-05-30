from typing import List

from rm_gallery.core.data.schema import DataSample


def load_rewardbench_test_samples(
    input_path: str, subsets: list[str] = None, sample_limit: int = -1
):
    rewardbench_path = input_path
    samples = []
    with open(rewardbench_path, "r") as file:
        for line in file.readlines():
            sample = DataSample.parse_raw(line.strip())
            # filter set
            sample_subset = sample.metadata["raw_data"]["subset"]
            if not subsets:
                samples.append(sample)
            if subsets and sample_subset in sample_subset:
                samples.append(sample)
            if 0 < sample_limit <= len(samples):
                break
    return samples


def write_datasamples_to_file(samples: List[DataSample], path: str):
    with open(path, "a") as writer:
        for sample in samples:
            writer.write(sample.model_dump_json())
            writer.write("\n")


# samples = load_rewardbench_test_samples()
# subsets = []
# for sample in samples:
#     sample_subset = sample.metadata["raw_data"]["subset"]
#     if sample_subset not in subsets:
#         subsets.append(sample_subset)
# logger.info(f"subsets={subsets}")
