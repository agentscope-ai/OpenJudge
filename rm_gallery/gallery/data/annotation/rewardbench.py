from typing import Any, Dict

from rm_gallery.core.data.annotation.template import (
    AnnotationTemplateRegistry,
    BaseAnnotationTemplate,
)


@AnnotationTemplateRegistry.register("rewardbench")
class RewardBenchAnnotationTemplate(BaseAnnotationTemplate):
    """Reward Bench annotation template implementation"""

    def __init__(self, name: str):
        super().__init__(name)

    @property
    def label_config(self) -> str:
        """Return the Label Studio XML configuration for reward bench evaluation"""
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

  <!-- Step 1: Ranking -->
  <View>
    <Text name="step1_title" value="Step 1: Answer Ranking" />
    <Text name="step1_desc1" value="Please select the most appropriate ranking relationship" />
    <Choices name="answer_ranking" toName="output_dialogue" choice="single" title="🏆 Answer Ranking">
      <Choice value="1>2" showIf="$answer_count=2"/>
      <Choice value="2>1" showIf="$answer_count=2"/>
      <Choice value="Neither" showIf="$answer_count=2"/>
      <Choice value="All answers are of equal quality"/>
    </Choices>
  </View>

  <!-- Step 2: Answer Rating -->
  <View>
    <Text name="step2_spacer" value="" />
    <Text name="step2_title" value="Step 2: Answer Rating" />
    <Text name="step2_desc" value="Please rate the quality of the answers for the $task_category task_category (1-5 stars)" />

    <Text name="answer1_label" value="📝 Answer 1 Rating:" />
    <Rating name="answer1_rating" toName="output_dialogue" maxRating="5" icon="star" size="medium" title="Answer 1 Quality Rating"/>

    <Text name="answer2_label" value="📝 Answer 2 Rating:" />
    <Rating name="answer2_rating" toName="output_dialogue" maxRating="5" icon="star" size="medium" title="Answer 2 Quality Rating"/>

    <Text name="rating_criteria" value="💡 Rating Criteria: 5 stars = excellent, 4 stars = good, 3 stars = average, 2 stars = poor, 1 star = very poor" />
  </View>

  <!-- Step 3: Additional Comments -->
  <View>
    <Text name="step3_spacer" value="" />
    <Text name="step3_title" value="Step 3: Additional Comments" />
    <Text name="step3_desc" value="Please provide any additional comments or feedback" />
    <TextArea name="additional_comments" toName="output_dialogue" placeholder="[x] The x-th answer has the following issues..." title="Additional Comments"/>
  </View>

</View>
"""

    def process_annotations(self, annotation_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process annotation data specific to reward bench evaluation

        Args:
            annotation_data: Generic annotation data with ratings, choices, text_areas

        Returns:
            Processed data structured for reward bench evaluation
        """
        processed = {
            "ranking_result": None,
            "answer_ratings": {},
            "quality_comparison": {},
            "comments": "",
            "preference": None,
        }

        # Extract answer ranking (Step 1)
        if "answer_ranking" in annotation_data.get("choices", {}):
            ranking_choices = annotation_data["choices"]["answer_ranking"]["choices"]
            if ranking_choices:
                processed["ranking_result"] = ranking_choices[0]

                # Determine preference based on ranking
                if "1>2" in ranking_choices[0]:
                    processed["preference"] = "answer_1"
                elif "2>1" in ranking_choices[0]:
                    processed["preference"] = "answer_2"
                elif "Neither" in ranking_choices[0]:
                    processed["preference"] = "neither"
                else:
                    processed["preference"] = "tie"

        # Extract answer ratings (Step 2)
        ratings = annotation_data.get("ratings", {})

        if "answer1_rating" in ratings:
            processed["answer_ratings"]["answer_1"] = ratings["answer1_rating"][
                "rating"
            ]

        if "answer2_rating" in ratings:
            processed["answer_ratings"]["answer_2"] = ratings["answer2_rating"][
                "rating"
            ]

        # Calculate quality comparison
        if len(processed["answer_ratings"]) == 2:
            rating1 = processed["answer_ratings"]["answer_1"]
            rating2 = processed["answer_ratings"]["answer_2"]

            processed["quality_comparison"] = {
                "rating_difference": rating1 - rating2,
                "better_answer": "answer_1"
                if rating1 > rating2
                else "answer_2"
                if rating2 > rating1
                else "tie",
                "rating_consistency": processed["preference"]
                == processed["quality_comparison"].get("better_answer", "unknown"),
            }

        # Extract additional comments (Step 3)
        if "additional_comments" in annotation_data.get("text_areas", {}):
            processed["comments"] = annotation_data["text_areas"][
                "additional_comments"
            ]["text"]

        return processed

    def validate_annotation_data(self, annotation_data: Dict[str, Any]) -> bool:
        """
        Validate annotation data for reward bench evaluation

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

        # Check if answer ranking is provided
        if "answer_ranking" not in annotation_data.get("choices", {}):
            return False

        # Check if at least one rating is provided
        ratings = annotation_data.get("ratings", {})
        if not any(key in ratings for key in ["answer1_rating", "answer2_rating"]):
            return False

        return True
