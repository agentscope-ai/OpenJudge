from typing import List

from rm_gallery.core.data.schema import DataSample


def calc_acc(samples: List[DataSample]):
    labels = []
    for sample in samples:
        for output in sample.output:
            if (
                output.answer.label["preference"] == "chosen"
                and output.answer.reward.details
            ):
                score = sum(r.score for r in output.answer.reward.details)
                if score > 0:
                    labels.append(1)
                else:
                    labels.append(0)

    return sum(labels) / len(labels)
