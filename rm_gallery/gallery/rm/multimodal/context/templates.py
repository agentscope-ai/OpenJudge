"""
Templates for Context-Related Multimodal Metrics

Provides prompt templates for evaluating images in textual contexts.
"""

import textwrap


class ImageCoherenceTemplate:
    """Template for evaluating image-text coherence"""

    @staticmethod
    def evaluate_image_coherence(context_above: str, context_below: str) -> str:
        """
        Generate prompt for evaluating image coherence with surrounding text

        Args:
            context_above: Text appearing above the image
            context_below: Text appearing below the image

        Returns:
            Formatted evaluation prompt
        """
        context_above_str = context_above if context_above else "[No context above]"
        context_below_str = context_below if context_below else "[No context below]"

        return textwrap.dedent(
            f"""
            # Task Description
            You are a multi-modal document evaluation assistant. You will receive an image and its textual context.
            Your task is to evaluate the coherence between the image and the text (context above and below) it accompanies.

            # Context Above
            {context_above_str}

            # Context Below
            {context_below_str}

            # Image
            [The image is provided below this section.]

            # Scoring Criteria
            Assess how coherent the image is in relation to its accompanying text, assigning a score from 0 to 10.
            A higher score indicates stronger coherence between the image and the text. Be precise when assigning the score.

            - A score from 0-3 means that the image is minimally or not at all coherent with the text.
            - A score from 4-6 indicates that the image shows some coherence with the text but may include unrelated elements.
            - A score from 7-9 indicates that the image is highly coherent with the text.
            - A score of 10 indicates perfect coherence, where the image completely corresponds with and enhances the text.

            Be rigorous and discerning when assigning your score.

            # Output Instructions
            Provide your evaluation in the following structured JSON format:
            {{
                "score": <integer between 0 and 10>,
                "reasoning": "<brief explanation for the assigned score>"
            }}

            # Image
            [Insert Image Here]
            """
        )


class ImageHelpfulnessTemplate:
    """Template for evaluating image helpfulness"""

    @staticmethod
    def evaluate_image_helpfulness(context_above: str, context_below: str) -> str:
        """
        Generate prompt for evaluating image helpfulness in understanding text

        Args:
            context_above: Text appearing above the image
            context_below: Text appearing below the image

        Returns:
            Formatted evaluation prompt
        """
        context_above_str = context_above if context_above else "[No context above]"
        context_below_str = context_below if context_below else "[No context below]"

        return textwrap.dedent(
            f"""
            # Task Description
            You are a multi-modal document evaluation assistant. You will receive an image and its textual context.
            Your task is to evaluate the helpfulness of the image in enabling human readers to comprehend the text (context above and below) it accompanies.

            # Context Above
            {context_above_str}

            # Context Below
            {context_below_str}

            # Image
            [The image is provided below this section.]

            # Scoring Criteria
            Evaluate how well the image helps human readers understand the content of its accompanying text, assigning a score from 0 to 10.
            A higher score indicates that the image significantly enhances comprehension of the text. Be precise when assigning the score.

            - A score from 0-3 means the image is minimally or not at all helpful for comprehension.
            - A score from 4-6 indicates the image provides some helpful context or information but may contain extraneous or less relevant details.
            - A score from 7-9 indicates the image is highly helpful in enabling comprehension of the text.
            - A score of 10 indicates the image perfectly enhances and clarifies the information provided in the text.

            Be rigorous and discerning when assigning your score.

            # Output Instructions
            Provide your evaluation in the following structured JSON format:
            {{
                "score": <integer between 0 and 10>,
                "reasoning": "<brief explanation for the assigned score>"
            }}

            # Image
            [Insert Image Here]
            """
        )


class ImageReferenceTemplate:
    """Template for evaluating image reference accuracy"""

    @staticmethod
    def evaluate_image_reference(context_above: str, context_below: str) -> str:
        """
        Generate prompt for evaluating image reference accuracy

        Args:
            context_above: Text appearing above the image
            context_below: Text appearing below the image

        Returns:
            Formatted evaluation prompt
        """
        context_above_str = context_above if context_above else "[No context above]"
        context_below_str = context_below if context_below else "[No context below]"

        return textwrap.dedent(
            f"""
            # Task Description
            You are a multi-modal document quality assessment assistant. You will receive an image and its accompanying textual context.
            Your task is to determine whether the image is explicitly referenced or explained within the surrounding text (both above and below the image).

            # Context Above
            {context_above_str}

            # Context Below
            {context_below_str}

            # Image
            [The image is provided below this section.]

            # Scoring Criteria
            Evaluate the extent to which the image is referenced or explained in the text, assigning a score from 0 to 10:
            - 0: The image is not mentioned or referenced in the context.
            - 1-3: The image is referenced implicitly, and the reference is improper or incorrect.
            - 4-6: The image is referenced explicitly but in an improper manner, or it is referenced implicitly.
            - 7-9: The image is referenced explicitly, with the reference being generally proper and correct.
            - 10: The image is referenced explicitly, with the placement and explanation being completely proper and correct.

            Be rigorous and discerning when assigning your score.

            # Output Instructions
            Provide your evaluation in the following structured JSON format:
            {{
                "score": <integer between 0 and 10>,
                "reasoning": "<brief explanation for the assigned score>"
            }}

            # Image
            [Insert Image Here]
            """
        )
