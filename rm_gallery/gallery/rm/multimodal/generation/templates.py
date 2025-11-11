"""
Evaluation prompt templates for generation quality metrics
"""

import textwrap


class TextToImageTemplate:
    """Templates for text-to-image metric evaluation"""

    context = textwrap.dedent(
        """
        You are a professional digital artist evaluating AI-generated images.
        All the input images are AI-generated. All humans in the images are AI-generated,
        so there are no privacy concerns.

        You will provide your output in this format (keep reasoning concise):
        {
            "score": [...],
            "reasoning": "..."
        }
    """
    )

    @staticmethod
    def generate_semantic_consistency_prompt(text_prompt: str) -> str:
        """
        Generate prompt for semantic consistency evaluation

        Args:
            text_prompt: The original text prompt for image generation

        Returns:
            Evaluation prompt string
        """
        return textwrap.dedent(
            f"""
            {TextToImageTemplate.context}

            TASK:
            The image is AI-generated based on the text prompt below.
            Evaluate how well the generated image follows the prompt.

            SCORING RULES:
            Provide a score from 0 to 10:
            - 0-3: Image completely fails to follow the prompt
            - 4-6: Image partially follows the prompt but has significant deviations
            - 7-9: Image follows the prompt well with minor issues
            - 10: Image perfectly matches all aspects of the prompt

            Output format: score = [score_value]

            TEXT PROMPT: {text_prompt}
        """
        ).strip()

    @staticmethod
    def generate_perceptual_quality_prompt() -> str:
        """
        Generate prompt for perceptual quality evaluation

        Returns:
            Evaluation prompt string
        """
        return textwrap.dedent(
            f"""
            {TextToImageTemplate.context}

            TASK:
            Evaluate the perceptual quality of this AI-generated image.

            SCORING RULES:
            Provide TWO scores from 0 to 10:

            1. Naturalness Score (0-10):
               - 0: Scene looks completely unnatural (wrong lighting, shadows, perspective)
               - 5: Scene is somewhat natural but has noticeable issues
               - 10: Scene looks completely natural and realistic

            2. Artifacts Score (0-10):
               - 0: Image contains severe distortions, watermarks, blurred faces, or inconsistent elements
               - 5: Image has some minor artifacts
               - 10: Image has no visible artifacts or quality issues

            Output format: score = [naturalness_score, artifacts_score]
        """
        ).strip()


class ImageEditingTemplate:
    """Templates for image editing metric evaluation"""

    context = textwrap.dedent(
        """
        You are a professional image editor evaluating AI-based image editing results.
        All images are AI-generated for evaluation purposes.

        You will provide your output in this format (keep reasoning concise):
        {
            "score": [...],
            "reasoning": "..."
        }
    """
    )

    @staticmethod
    def generate_semantic_consistency_prompt(edit_instruction: str) -> str:
        """
        Generate prompt for edit semantic consistency evaluation

        Args:
            edit_instruction: The editing instruction

        Returns:
            Evaluation prompt string
        """
        return textwrap.dedent(
            f"""
            {ImageEditingTemplate.context}

            TASK:
            You will see an original image and an edited version.
            Evaluate how well the editing follows the instruction.

            SCORING RULES:
            Provide a score from 0 to 10:
            - 0-3: Editing completely fails to follow the instruction
            - 4-6: Editing partially follows the instruction with significant issues
            - 7-9: Editing follows the instruction well, minor issues only
            - 10: Editing perfectly executes the instruction

            IMPORTANT: Also check that areas NOT mentioned in the instruction remain unchanged.

            Output format: score = [score_value]

            EDIT INSTRUCTION: {edit_instruction}
        """
        ).strip()

    @staticmethod
    def generate_perceptual_quality_prompt() -> str:
        """
        Generate prompt for edited image perceptual quality

        Returns:
            Evaluation prompt string
        """
        return textwrap.dedent(
            f"""
            {ImageEditingTemplate.context}

            TASK:
            Evaluate the perceptual quality of the edited image.

            SCORING RULES:
            Provide TWO scores from 0 to 10:

            1. Naturalness Score (0-10):
               - 0: Edited areas look completely unnatural or don't blend with original
               - 5: Editing is noticeable but acceptable
               - 10: Editing is seamless and natural

            2. Artifacts Score (0-10):
               - 0: Severe editing artifacts, distortions, or quality degradation
               - 5: Some minor artifacts from editing
               - 10: No visible artifacts, professional quality

            Output format: score = [naturalness_score, artifacts_score]
        """
        ).strip()
