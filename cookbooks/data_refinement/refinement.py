import asyncio
from abc import ABC
from typing import Callable, List

from openjudge.graders.base_grader import BaseGrader
from openjudge.graders.schema import GraderMode, GraderResult
from openjudge.models.base_chat_model import BaseChatModel
from openjudge.models.schema.oai.message import ChatMessage
from openjudge.models.schema.oai.response import ChatResponse
from openjudge.models.schema.prompt_template import PromptTemplate
from openjudge.utils.mapping import parse_data_with_mapper


def format_messages(messages: List[ChatMessage]) -> str:
    """
        Formats chat messages into XML-style string representation.

        Takes a list of ChatMessage objects and converts them into a structured
        XML-like format where each message is wrapped in tags corresponding to
        its role (e.g., <user>, <assistant>).

        Args:
            messages (List[ChatMessage]): List of ChatMessage objects to format.
                Each ChatMessage should have 'role' and 'content' attributes.

        Returns:
            str: Formatted string with messages wrapped in role-specific tags.
                 Returns empty string if messages list is empty.

        Example:
            >>> messages = [
            ...     ChatMessage(role="user", content="Hello!"),
            ...     ChatMessage(role="assistant", content="Hi there!")
            ... ]
            >>> formatted = format_messages(messages)
            >>> print(formatted)
            "<user>Hello!</user>
    <assistant>Hi there!</assistant>"
    """
    return "\n".join(
        [f"<{message.role}>{message.content}</{message.role}>" for message in messages],
    )


GENERATE_RESPONSE_PROMPT = """# Task
Please generate a response as the conversation required.

# Conversation history
{history}
"""


REFINE_RESPONSE_PROMPT = """# Task
Please generate a better response based on the feedback provided on candidate responses.

# Conversation history
{history}

# Responses
{responses}

# Feedback
{feedback}
"""


GENERATE_RESPONSE_TEMPLATE = PromptTemplate(
    messages=[
        ChatMessage(
            role="user",
            content=GENERATE_RESPONSE_PROMPT,
        ),
    ],
)


REFINE_RESPONSE_TEMPLATE = PromptTemplate(
    messages=[
        ChatMessage(
            role="user",
            content=REFINE_RESPONSE_PROMPT,
        ),
    ],
)


class LLMRefinement(ABC):
    """
    A module for iterative response refinement using LLM and reward feedback mechanisms.

    This module implements a refinement process where language models iteratively improve
    their responses based on feedback from a grading system. It supports both initial
    response generation and subsequent refinements through multiple iterations.

    Example:
        >>> from openjudge.graders import LLMPairwiseGrader
        >>> from openjudge.models import QwenChatModel
        >>> from .refinement import LLMRefinement
        >>>
        >>> grader = LLMPairwiseGrader()
        >>> model = OpenAIChatModel(model="qwen3-max")
        >>> refiner = LLMRefinement(grader, model, max_iterations=3)
        >>>
        >>> sample = {
        ...     "history": [
        ...         {"role": "user", "content": "How do I sort a list in Python?"},
        ...     ]
        ... }
        >>> refined_sample = refiner.refine(sample)
    """

    _generate_response_template = GENERATE_RESPONSE_TEMPLATE
    _refine_response_template = REFINE_RESPONSE_TEMPLATE

    def __init__(self, grader: BaseGrader, model: BaseChatModel, max_iterations: int = 3) -> None:
        """
        Initialize the LLMRefinement instance.

        Args:
            grader (BaseGrader): The grader used to evaluate response quality and provide feedback.
                Must be a listwise grader (grader.mode == GraderMode.LISTWISE).
            model (BaseChatModel): The language model used for generating responses.
            max_iterations (int, optional): Maximum number of refinement iterations. Defaults to 3.

        Raises:
            AssertionError: If the provided grader is not a listwise grader (grader.mode != GraderMode.LISTWISE).

        Example:
            >>> from openjudge.graders import LLMPairwiseGrader
            >>> from openjudge.models import QwenChatModel
            >>>
            >>> grader = LLMPairwiseGrader()
            >>> model = OpenAIChatModel(model="qwen3-max")
            >>> refiner = LLMRefinement(grader, model, max_iterations=5)
        """
        self.grader = grader
        assert self.grader.mode == GraderMode.LISTWISE
        self.model = model
        self.max_iterations = max_iterations

    def _generate_response(
        self,
        sample: dict,
        **kwargs,
    ) -> dict:
        """
        Generate initial response based on conversation history.

        This method generates the initial response using the language model based on
        the conversation history provided in the sample. It formats the history into
        a structured prompt and uses the model to generate a response.

        Note: The feedback parameter is not currently used in this implementation but
        is kept for API consistency with other methods.

        Args:
            sample (dict): Sample data containing conversation history. Expected to have
                a "history" key with a list of message dictionaries.
            **kwargs: Additional parameters passed to the language model generation method.

        Returns:
            dict: Updated sample with the generated response appended to the "responses" list.
                The response is stored as a dictionary with model response attributes.

        Example:
            >>> sample = {
            ...     "history": [
            ...         {"role": "user", "content": "Hello, how are you?"},
            ...     ]
            ... }
            >>> updated_sample = self._generate_response(sample)
        """
        history = format_messages(sample.get("history", []))
        messages = self._generate_response_template.format(history=history)
        response = asyncio.run(self.model.achat(messages=messages))
        sample["responses"] = []
        sample["responses"].append(
            response.model_dump(),
        )
        return sample

    def _refine_response(self, sample: dict, feedback: str, **kwargs) -> dict:
        """
        Refine response based on conversation history and feedback.

        This method generates an improved response by incorporating feedback on previous
        responses. It formats the conversation history, previous responses, and feedback
        into a structured prompt and uses the model to generate a refined response.

        Args:
            sample (dict): Sample data containing conversation history and previous responses.
                Expected to have a "history" key with message list and "responses" key with
                previous response dictionaries.
            feedback (str): Quality assessment feedback for previous responses that guides
                the refinement process.
            **kwargs: Additional parameters passed to the language model generation method.

        Returns:
            dict: Updated sample with the refined response appended to the "responses" list.
                The response is stored as a dictionary with model response attributes.

        Example:
            >>> sample = {
            ...     "history": [
            ...         {"role": "user", "content": "Hello, how are you?"},
            ...     ],
            ...     "responses": [
            ...         {"content": "I'm fine, thank you! How can I help you today?"}
            ...     ]
            ... }
            >>> feedback = "The response is good but could be more empathetic."
            >>> updated_sample = self._refine_response(sample, feedback)
        """
        history = format_messages(sample.get("history", []))
        responses = "\n".join(
            [
                f"<response_{i}>{content}</response_{i}>"
                for i, content in enumerate([response["content"] for response in sample["responses"]])
            ],
        )
        messages = self._refine_response_template.format(history=history, responses=responses, feedback=feedback)
        response = asyncio.run(self.model.achat(messages=messages))
        sample["responses"].append(
            response.model_dump(),
        )
        return sample

    def _revise_response(self, sample: dict, **kwargs) -> str:
        """
        Generate quality feedback for a response sample using the grader.

        This method evaluates the latest response in the sample using the configured grader
        and returns feedback that can guide the refinement process. The grader assesses the
        response quality and provides a reason/feedback string.

        Args:
            sample (dict): Data sample containing input-response pairs for evaluation.
                Expected to have a "responses" key with a list of response dictionaries.
            **kwargs: Additional parameters passed to the grader evaluation method.

        Returns:
            str: Feedback string describing response quality assessment. This feedback
                is used to guide the next refinement iteration.

        Example:
            >>> sample = {
            ...     "history": [
            ...         {"role": "user", "content": "Hello, how are you?"},
            ...     ],
            ...     "responses": [
            ...         {"content": "I'm fine, thank you! How can I help you today?"}
            ...     ]
            ... }
            >>> feedback = self._revise_response(sample)
        """
        # Evaluate response quality using reward module
        grader_rank = asyncio.run(self.grader.aevaluate(**sample))
        feedback = grader_rank.reason
        return feedback

    def refine(self, sample: dict, mapper: dict | Callable | None = None, **kwargs) -> dict:
        """
        Execute iterative response refinement process.

        This method performs iterative refinement of responses using the following process:
        1. Generate initial response if none exists
        2. For each iteration up to max_iterations:
           a. Evaluate latest response using grader to get feedback
           b. Generate refined response based on feedback

        The refinement process improves response quality progressively by incorporating
        automated feedback in each iteration.

        Args:
            sample (dict): Data sample containing input for refinement. Should contain
                a "history" key with conversation history. Optionally may include
                existing "responses" to continue refinement.
            mapper (dict/callable): Mapping relationship, key is path, value is field name or callable
            **kwargs: Additional parameters passed to generation and evaluation methods.

        Returns:
            dict: Final refined sample with all responses in the "responses" list.
                The last entry in the responses list contains the final refined response.

        Example:
            >>> from openjudge.graders import LLMPairwiseGrader
            >>> from openjudge.models import QwenChatModel
            >>> from .refinement import LLMRefinement
            >>>
            >>> grader = LLMPairwiseGrader()
            >>> model = OpenAIChatModel(model="qwen3-max")
            >>> refiner = LLMRefinement(grader, model)
            >>>
            >>> sample = {
            ...     "history": [
            ...         {"role": "user", "content": "Explain quantum computing in simple terms"},
            ...     ]
            ... }
            >>> refined_sample = refiner.refine(sample)
            >>> print(refined_sample["responses"][-1]["content"])
            # Outputs the final refined response
        """
        sample = parse_data_with_mapper(sample, mapper)
        if len(sample.get("responses", [])) == 0:
            # Initial response generation
            sample = self._generate_response(sample, **kwargs)

        # Iterative refinement loop
        for i in range(self.max_iterations):
            # Generate feedback and create refined response
            feedback = self._revise_response(sample, **kwargs)
            sample = self._refine_response(sample, feedback, **kwargs)

        return sample


if __name__ == "__main__":
    # Mock grader for demonstration purposes
    class MockListwiseGrader(BaseGrader):
        def __init__(self):
            super().__init__(
                name="mock_listwise_grader",
                mode=GraderMode.LISTWISE,
                description="A mock listwise grader for demonstration",
            )

        async def aevaluate(self, **kwargs) -> GraderResult:
            # Simulate some evaluation logic
            responses = kwargs.get("responses", [])
            if responses:
                # Return a mock result with feedback
                return GraderResult(
                    name=self.name,
                    reason="The response could be more detailed and provide examples.",
                    score=0.7,
                    rank=[1],
                )
            return GraderResult(
                name=self.name,
                reason="Initial response needs improvement.",
                score=0.5,
                rank=[1],
            )

    # Mock model for demonstration purposes
    class MockChatModel(BaseChatModel):
        def __init__(self):
            super().__init__(model="mock-model", stream=False)

        async def achat(self, messages, **kwargs):
            # Simulate a model response
            return ChatResponse(
                content="This is a simulated model response.",
            )

    # Example usage of LLMRefinement
    def main():
        # Initialize a mock grader
        grader = MockListwiseGrader()

        # Initialize a mock model
        model = MockChatModel()

        # Create the refiner
        refiner = LLMRefinement(grader, model, max_iterations=2)

        # Sample data for refinement
        sample = {
            "history": [
                {"role": "user", "content": "Explain quantum computing in simple terms."},
            ],
        }

        # Run the refinement process
        print("Starting refinement process...")
        refined_sample = refiner.refine(sample)
        print("Refinement completed.")
        print(f"Number of responses generated: {len(refined_sample['responses'])}")
        print("Final response:")
        print(refined_sample["responses"][-1])

    # Run the example
    main()
