from typing import Any, Dict

from rm_gallery.core.data.annotation.template import (
    AnnotationTemplateRegistry,
    BaseAnnotationTemplate,
)


@AnnotationTemplateRegistry.register("rewardbench2")
class RewardBench2AnnotationTemplate(BaseAnnotationTemplate):
    """Reward Bench 2 annotation template implementation for 4-way comparison"""

    def __init__(self, name: str):
        super().__init__(name)

    @property
    def label_config(self) -> str:
        """Return the Label Studio XML configuration for reward bench 2 evaluation (4-way comparison)"""
        return """
<View>
  <!-- Sample Information -->
  <Header value="Sample Information"/>
  <Text name="unique_id" value="$unique_id" title="Unique ID"/>
  <Text name="source" value="$source" title="Source"/>
  <Text name="task_category" value="$task_category" title="task_category"/>
  <Text name="created_at" value="$created_at" title="Created At"/>
  <Text name="answer_count" value="$answer_count" title="Number of Answers"/>

  <!-- Input Messages -->
  <Header value="Input Messages"/>
  <Paragraphs name="input_dialogue" value="$input_messages" layout="dialogue" nameKey="role" textKey="content" />

  <!-- Output Responses -->
  <Header value="Output Responses"/>
  <Paragraphs name="output_dialogue" value="$output_messages" layout="dialogue" nameKey="role" textKey="content" />

  <!-- Step 1: Best Answer Selection -->
  <View>
    <Text name="step1_title" value="Step 1: Best Answer Selection" />
    <Text name="step1_desc1" value="Please select the best answer among the 4 options" />
    <Choices name="best_answer" toName="output_dialogue" choice="single" title="🏆 Best Answer">
      <Choice value="answer_1" showIf="$answer_count>=1"/>
      <Choice value="answer_2" showIf="$answer_count>=2"/>
      <Choice value="answer_3" showIf="$answer_count>=3"/>
      <Choice value="answer_4" showIf="$answer_count>=4"/>
      <Choice value="all_equal" showIf="$answer_count=4"/>
    </Choices>
  </View>

  <!-- Step 2: Answer Ranking -->
  <View>
    <Text name="step2_spacer" value="" />
    <Text name="step2_title" value="Step 2: Answer Ranking" />
    <Text name="step2_desc" value="Please rank all answers from best to worst (1=best, 4=worst)" />

    <Text name="answer1_rank_label" value="📝 Answer 1 Rank:" />
    <Choices name="answer1_rank" toName="output_dialogue" choice="single" title="Answer 1 Rank">
      <Choice value="1"/>
      <Choice value="2"/>
      <Choice value="3"/>
      <Choice value="4"/>
    </Choices>

    <Text name="answer2_rank_label" value="📝 Answer 2 Rank:" />
    <Choices name="answer2_rank" toName="output_dialogue" choice="single" title="Answer 2 Rank">
      <Choice value="1"/>
      <Choice value="2"/>
      <Choice value="3"/>
      <Choice value="4"/>
    </Choices>

    <Text name="answer3_rank_label" value="📝 Answer 3 Rank:" />
    <Choices name="answer3_rank" toName="output_dialogue" choice="single" title="Answer 3 Rank">
      <Choice value="1"/>
      <Choice value="2"/>
      <Choice value="3"/>
      <Choice value="4"/>
    </Choices>

    <Text name="answer4_rank_label" value="📝 Answer 4 Rank:" />
    <Choices name="answer4_rank" toName="output_dialogue" choice="single" title="Answer 4 Rank">
      <Choice value="1"/>
      <Choice value="2"/>
      <Choice value="3"/>
      <Choice value="4"/>
    </Choices>
  </View>

  <!-- Step 3: Answer Rating -->
  <View>
    <Text name="step3_spacer" value="" />
    <Text name="step3_title" value="Step 3: Answer Rating" />
    <Text name="step3_desc" value="Please rate the quality of each answer for the $task_category task_category (1-5 stars)" />

    <Text name="answer1_rating_label" value="📝 Answer 1 Rating:" />
    <Rating name="answer1_rating" toName="output_dialogue" maxRating="5" icon="star" size="medium" title="Answer 1 Quality Rating"/>

    <Text name="answer2_rating_label" value="📝 Answer 2 Rating:" />
    <Rating name="answer2_rating" toName="output_dialogue" maxRating="5" icon="star" size="medium" title="Answer 2 Quality Rating"/>

    <Text name="answer3_rating_label" value="📝 Answer 3 Rating:" />
    <Rating name="answer3_rating" toName="output_dialogue" maxRating="5" icon="star" size="medium" title="Answer 3 Quality Rating"/>

    <Text name="answer4_rating_label" value="📝 Answer 4 Rating:" />
    <Rating name="answer4_rating" toName="output_dialogue" maxRating="5" icon="star" size="medium" title="Answer 4 Quality Rating"/>

    <Text name="rating_criteria" value="💡 Rating Criteria: 5 stars = excellent, 4 stars = good, 3 stars = average, 2 stars = poor, 1 star = very poor" />
  </View>

  <!-- Step 4: Additional Comments -->
  <View>
    <Text name="step4_spacer" value="" />
    <Text name="step4_title" value="Step 4: Additional Comments" />
    <Text name="step4_desc" value="Please provide any additional comments or feedback" />
    <TextArea name="additional_comments" toName="output_dialogue" placeholder="[x] The x-th answer has the following issues..." title="Additional Comments"/>
  </View>

</View>
"""

    def process_annotations(self, annotation_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process annotation data specific to reward bench 2 evaluation (4-way comparison)

        Args:
            annotation_data: Generic annotation data with ratings, choices, text_areas

        Returns:
            Processed data structured for reward bench 2 evaluation
        """
        processed = {
            "best_answer": None,
            "answer_rankings": {},
            "answer_ratings": {},
            "ranking_order": [],
            "quality_comparison": {},
            "comments": "",
            "preference": None,
        }

        # Extract best answer selection (Step 1)
        if "best_answer" in annotation_data.get("choices", {}):
            best_answer_choices = annotation_data["choices"]["best_answer"]["choices"]
            if best_answer_choices:
                processed["best_answer"] = best_answer_choices[0]
                processed["preference"] = best_answer_choices[0]

        # Extract answer rankings (Step 2)
        choices = annotation_data.get("choices", {})
        rank_keys = ["answer1_rank", "answer2_rank", "answer3_rank", "answer4_rank"]

        for i, rank_key in enumerate(rank_keys, 1):
            if rank_key in choices:
                rank_choices = choices[rank_key]["choices"]
                if rank_choices:
                    processed["answer_rankings"][f"answer_{i}"] = int(rank_choices[0])

        # Create ranking order based on ranks
        if processed["answer_rankings"]:
            # Sort answers by their rank (1=best, 4=worst)
            sorted_answers = sorted(
                processed["answer_rankings"].items(), key=lambda x: x[1]
            )
            processed["ranking_order"] = [answer for answer, rank in sorted_answers]

        # Extract answer ratings (Step 3)
        ratings = annotation_data.get("ratings", {})
        rating_keys = [
            "answer1_rating",
            "answer2_rating",
            "answer3_rating",
            "answer4_rating",
        ]

        for i, rating_key in enumerate(rating_keys, 1):
            if rating_key in ratings:
                processed["answer_ratings"][f"answer_{i}"] = ratings[rating_key][
                    "rating"
                ]

        # Calculate quality comparison
        if processed["answer_ratings"]:
            # Find the highest rated answer
            best_rated_answer = max(
                processed["answer_ratings"].items(), key=lambda x: x[1]
            )

            # Calculate average rating
            avg_rating = sum(processed["answer_ratings"].values()) / len(
                processed["answer_ratings"]
            )

            processed["quality_comparison"] = {
                "best_rated_answer": best_rated_answer[0],
                "best_rating": best_rated_answer[1],
                "average_rating": avg_rating,
                "rating_spread": max(processed["answer_ratings"].values())
                - min(processed["answer_ratings"].values()),
                "consistency_check": {
                    "best_answer_matches_best_rating": processed["best_answer"]
                    == best_rated_answer[0],
                    "best_answer_matches_rank_1": processed["best_answer"]
                    in [
                        ans
                        for ans, rank in processed["answer_rankings"].items()
                        if rank == 1
                    ]
                    if processed["answer_rankings"]
                    else False,
                },
            }

        # Extract additional comments (Step 4)
        if "additional_comments" in annotation_data.get("text_areas", {}):
            processed["comments"] = annotation_data["text_areas"][
                "additional_comments"
            ]["text"]

        return processed

    def validate_annotation_data(self, annotation_data: Dict[str, Any]) -> bool:
        """
        Validate annotation data for reward bench 2 evaluation

        Args:
            annotation_data: Annotation data to validate

        Returns:
            True if valid, False otherwise
        """
        # Check if required fields are present
        required_sections = ["choices", "ratings"]
        for section in required_sections:
            if section not in annotation_data:
                return False

        # Check if best answer is selected
        if "best_answer" not in annotation_data.get("choices", {}):
            return False

        # Check if at least some rankings are provided
        choices = annotation_data.get("choices", {})
        rank_keys = ["answer1_rank", "answer2_rank", "answer3_rank", "answer4_rank"]
        if not any(key in choices for key in rank_keys):
            return False

        # Check if at least some ratings are provided
        ratings = annotation_data.get("ratings", {})
        rating_keys = [
            "answer1_rating",
            "answer2_rating",
            "answer3_rating",
            "answer4_rating",
        ]
        if not any(key in ratings for key in rating_keys):
            return False

        # Validate ranking consistency (each rank should be unique)
        provided_ranks = []
        for rank_key in rank_keys:
            if rank_key in choices:
                rank_choices = choices[rank_key]["choices"]
                if rank_choices:
                    rank = int(rank_choices[0])
                    if rank in provided_ranks:
                        return False  # Duplicate rank
                    provided_ranks.append(rank)

        return True
