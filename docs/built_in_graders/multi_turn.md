# Multi-turn Conversation Graders

Graders for evaluating AI assistant capabilities in multi-turn conversations. These graders assess various aspects of dialogue quality including context memory, reference resolution, topic handling, and interaction patterns.


## Overview

| Grader | Purpose | Type | Score Range | Key Use Case |
|--------|---------|------|-------------|--------------|
| `ContextMemoryGrader` | Evaluates recall of earlier conversation details | LLM-Based | 1-5 | Long conversations |
| `AnaphoraResolutionGrader` | Evaluates pronoun and reference resolution | LLM-Based | 1-5 | Reference understanding |
| `TopicSwitchGrader` | Evaluates handling of sudden topic changes | LLM-Based | 1-5 | Multi-topic dialogues |
| `SelfCorrectionGrader` | Evaluates error correction based on feedback | LLM-Based | 1-5 | Error handling |
| `InstructionClarificationGrader` | Evaluates ability to ask for clarification | LLM-Based | 1-5 | Ambiguous queries |
| `ProactiveInteractionGrader` | Evaluates proactive engagement in conversation | LLM-Based | 1-5 | Chatbot quality |
| `ResponseRepetitionGrader` | Detects repetitive content in responses | LLM-Based | 1-5 | Response quality |


## Input Format

All multi-turn graders accept the same input format:

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `history` | List[Dict] | Yes | Conversation history in ChatMessage format |
| `response` | str | Yes | The assistant response to evaluate |

**History Format:**
```python
history = [
    {"role": "user", "content": "User message 1"},
    {"role": "assistant", "content": "Assistant response 1"},
    {"role": "user", "content": "User message 2"},
    # ... more turns
]
```


## ContextMemoryGrader

Evaluates whether the assistant can recall and utilize details from earlier parts of the conversation, maintaining content coherence.

**When to use:**
- Long multi-turn conversations
- Conversations with user preferences or constraints
- Testing context retention capabilities
- Quality assurance for chatbots

**Grading Criteria:**
- **5**: Perfect memory - all relevant details accurately remembered
- **4**: Good memory - most key information retained
- **3**: Basic memory - main information remembered with some gaps
- **2**: Insufficient memory - multiple important details forgotten
- **1**: Severe forgetting - response disconnected from context

**Example:**

```python
import asyncio
from openjudge.models import OpenAIChatModel
from openjudge.graders.multi_turn import ContextMemoryGrader

async def main():
    model = OpenAIChatModel(model="qwen-max")
    grader = ContextMemoryGrader(model=model)

    result = await grader.aevaluate(
        history=[
            {"role": "user", "content": "I'm allergic to nuts and prefer vegetarian food."},
            {"role": "assistant", "content": "I'll remember that for recommendations."},
            {"role": "user", "content": "What should I order for dinner?"},
        ],
        response="I recommend the walnut pesto pasta!",  # Forgot nut allergy!
    )

    print(f"Score: {result.score}")
    print(f"Reason: {result.reason}")

asyncio.run(main())
```

**Output:**

```
Score: 1.0
Reason: The assistant failed to remember the user's allergy to nuts, as evidenced by recommending a dish (walnut pesto pasta) that contains walnuts. While the recommendation is vegetarian, which aligns with one of the user's preferences, it completely disregards the critical information about the nut allergy, which could have serious health implications. This oversight significantly impacts the quality and safety of the response.
```


## AnaphoraResolutionGrader

Evaluates whether the assistant can accurately identify and resolve pronouns (e.g., "it", "this", "that", "they") to their correct referents.

**When to use:**
- Testing reference understanding
- Evaluating comprehension in complex dialogues
- Quality assurance for conversational AI

**Grading Criteria:**
- **5**: Perfect resolution - all referents correctly understood
- **4**: Good resolution - main references correct with minor deviations
- **3**: Basic resolution - some misunderstandings present
- **2**: Insufficient resolution - key pronouns misunderstood
- **1**: Severe errors - completely wrong referent identification

**Example:**

```python
import asyncio
from openjudge.models import OpenAIChatModel
from openjudge.graders.multi_turn import AnaphoraResolutionGrader

async def main():
    model = OpenAIChatModel(model="qwen-max")
    grader = AnaphoraResolutionGrader(model=model)

    result = await grader.aevaluate(
        history=[
            {"role": "user", "content": "I bought a laptop and a phone yesterday."},
            {"role": "assistant", "content": "Nice purchases!"},
            {"role": "user", "content": "Is it good for programming?"},  # "it" = laptop
        ],
        response="Yes, the laptop is excellent for programming.",
    )

    print(f"Score: {result.score}")
    print(f"Reason: {result.reason}")

asyncio.run(main())
```

**Output:**

```
Score: 5.0
Reason: The assistant correctly identified that 'it' in the user's question refers to one of the items mentioned earlier, and reasonably inferred that the laptop (rather than the phone) is more likely to be used for programming. The response directly addresses the laptop as being good for programming, which aligns with the user's intent. There were no reference errors, and the answer was both relevant and accurate.
```


## TopicSwitchGrader

Evaluates whether the assistant can recognize sudden topic changes and appropriately focus on the new topic without mixing information from previous topics.

**When to use:**
- Multi-topic conversations
- Testing context management
- Evaluating conversation flexibility

**Grading Criteria:**
- **5**: Perfect switch - accurate detection, focused response
- **4**: Good switch - correct handling with minor remnants
- **3**: Basic switch - some confusion or lack of focus
- **2**: Insufficient switch - mixed old topic content
- **1**: Failed switch - ignored topic change or severe confusion

**Example:**

```python
import asyncio
from openjudge.models import OpenAIChatModel
from openjudge.graders.multi_turn import TopicSwitchGrader

async def main():
    model = OpenAIChatModel(model="qwen-max")
    grader = TopicSwitchGrader(model=model)

    result = await grader.aevaluate(
        history=[
            {"role": "user", "content": "What's the best programming language?"},
            {"role": "assistant", "content": "Python is great for beginners."},
            {"role": "user", "content": "By the way, what's the weather in Tokyo?"},  # Topic switch!
        ],
        response="I don't have real-time weather data, but you can check weather.com.",
    )

    print(f"Score: {result.score}")
    print(f"Reason: {result.reason}")

asyncio.run(main())
```

**Output:**

```
Score: 5.0
Reason: The assistant correctly identified that the user switched from asking about programming languages to inquiring about the weather in Tokyo. The response provided is focused entirely on the new topic, offering a practical solution for the user to find the information they need. There is no confusion or mixing of information from the previous topic, and the transition is handled smoothly and naturally.
```


## SelfCorrectionGrader

Evaluates whether the assistant can appropriately acknowledge errors and provide correct information when users point out mistakes.

**When to use:**
- Testing error handling capabilities
- Evaluating response to user feedback
- Quality assurance for knowledge-based systems

**Grading Criteria:**
- **5**: Perfect correction - acknowledged, accurate, clear, professional
- **4**: Good correction - mostly accurate with minor gaps
- **3**: Basic correction - incomplete or attitude issues
- **2**: Insufficient correction - reluctant or still problematic
- **1**: Refused correction - insisted on error or defensive

**Example:**

```python
import asyncio
from openjudge.models import OpenAIChatModel
from openjudge.graders.multi_turn import SelfCorrectionGrader

async def main():
    model = OpenAIChatModel(model="qwen-max")
    grader = SelfCorrectionGrader(model=model)

    result = await grader.aevaluate(
        history=[
            {"role": "user", "content": "What's the capital of Australia?"},
            {"role": "assistant", "content": "The capital is Sydney."},  # Wrong!
            {"role": "user", "content": "I think that's incorrect."},
        ],
        response="You're right, I apologize. The capital of Australia is Canberra, not Sydney.",
    )

    print(f"Score: {result.score}")
    print(f"Reason: {result.reason}")

asyncio.run(main())
```

**Output:**

```
Score: 5.0
Reason: The assistant appropriately recognized the error in stating Sydney as the capital of Australia. The correction was made accurately, identifying Canberra as the correct capital. The assistant apologized for the mistake, which shows a professional and humble attitude towards the error. The explanation, while brief, was clear and to the point. There was no deflection or reluctance in acknowledging the mistake, and the corrected information was accurate. This response demonstrates all the elements required for a perfect self-correction.
```


## InstructionClarificationGrader

Evaluates whether the assistant can appropriately ask for clarification when faced with vague or incomplete queries.

**When to use:**
- Testing ambiguity handling
- Evaluating user guidance quality
- Quality assurance for customer service bots

**Grading Criteria:**
- **5**: Perfect clarification - accurate detection, targeted questions
- **4**: Good clarification - mostly appropriate with minor gaps
- **3**: Basic clarification - questions asked but not precise
- **2**: Insufficient clarification - failed to identify ambiguity
- **1**: Failed clarification - ignored ambiguity or over-questioned

**Example:**

```python
import asyncio
from openjudge.models import OpenAIChatModel
from openjudge.graders.multi_turn import InstructionClarificationGrader

async def main():
    model = OpenAIChatModel(model="qwen-max")
    grader = InstructionClarificationGrader(model=model)

    result = await grader.aevaluate(
        history=[
            {"role": "user", "content": "Book me a flight."},  # Vague query
        ],
        response="I'd be happy to help! Could you tell me your departure city, destination, and travel dates?",
    )

    print(f"Score: {result.score}")
    print(f"Reason: {result.reason}")

asyncio.run(main())
```

**Output:**

```
Score: 5.0
Reason: The assistant correctly identified that the user's request was vague and required additional information to proceed. The response asked for key missing details: departure city, destination, and travel dates, which are all essential for booking a flight. The questions were specific, targeted, and presented in a polite and professional manner. There was no over-questioning, and the assistant effectively guided the user to provide the necessary information.
```


## ProactiveInteractionGrader

Evaluates whether the assistant can proactively engage in conversation, maintain dialogue rhythm, and guide the conversation forward naturally.

**When to use:**
- Evaluating conversational engagement quality
- Testing dialogue management capabilities
- Quality assurance for chatbots

**Grading Criteria:**
- **5**: Perfect interaction - natural, relevant, and effective engagement
- **4**: Good interaction - mostly natural with minor stiffness
- **3**: Basic interaction - some proactive elements but not natural
- **2**: Insufficient interaction - mainly passive responses
- **1**: No interaction - completely passive and mechanical

**Example:**

```python
import asyncio
from openjudge.models import OpenAIChatModel
from openjudge.graders.multi_turn import ProactiveInteractionGrader

async def main():
    model = OpenAIChatModel(model="qwen-max")
    grader = ProactiveInteractionGrader(model=model)

    result = await grader.aevaluate(
        history=[
            {"role": "user", "content": "I'm planning a trip to Japan."},
        ],
        response="Japan is wonderful! When are you planning to go? I can suggest activities based on the season.",
    )

    print(f"Score: {result.score}")
    print(f"Reason: {result.reason}")

asyncio.run(main())
```

**Output:**

```
Score: 5.0
Reason: The assistant's response is a good example of proactive interaction. It naturally follows up on the user's statement about planning a trip to Japan by asking for more specific information (the travel dates), which shows attention to the user's needs and an effort to provide personalized advice. The suggestion to tailor activities based on the season also demonstrates a proactive approach in offering valuable supplementary information, aiming to enhance the user's experience. This not only keeps the conversation flowing but also guides it towards a meaningful direction that could be very helpful for the user. The response feels natural and appropriately timed, without being abrupt or excessive.
```


## ResponseRepetitionGrader

Evaluates whether the assistant's response contains repetitive content compared to earlier responses in the conversation.

**When to use:**
- Detecting circular dialogue patterns
- Assessing conversation efficiency
- Quality assurance for chatbot responses

**Grading Criteria:**
- **5**: No repetition - entirely new and valuable content
- **4**: Minimal repetition - mostly new information
- **3**: Partial repetition - some new useful information
- **2**: Significant repetition - very little new information
- **1**: Severe repetition - no new value provided

**Example:**

```python
import asyncio
from openjudge.models import OpenAIChatModel
from openjudge.graders.multi_turn import ResponseRepetitionGrader

async def main():
    model = OpenAIChatModel(model="qwen-max")
    grader = ResponseRepetitionGrader(model=model)

    result = await grader.aevaluate(
        history=[
            {"role": "user", "content": "What is Python?"},
            {"role": "assistant", "content": "Python is a high-level programming language."},
            {"role": "user", "content": "Tell me more."},
        ],
        response="Python is a high-level programming language.",  # Exact repeat!
    )

    print(f"Score: {result.score}")
    print(f"Reason: {result.reason}")

asyncio.run(main())
```

**Output:**

```
Score: 1.0
Reason: The current response 'Python is a high-level programming language.' is an exact repetition of the assistant's previous answer. This verbatim repetition does not provide any new information or advance the conversation, which can lead to a poor user experience and inefficiency in communication.
```


## Step-Level Evaluation

For comprehensive session evaluation, split a multi-turn conversation into steps and evaluate each assistant response separately, then aggregate scores.

**Example:**

```python
import asyncio
from openjudge.models import OpenAIChatModel
from openjudge.graders.multi_turn import ContextMemoryGrader

def split_session_to_steps(messages):
    """Split session into (history, response) tuples for each assistant turn."""
    steps = []
    history = []
    for msg in messages:
        if msg["role"] == "assistant":
            steps.append((list(history), msg["content"]))
        history.append(msg)
    return steps

async def main():
    model = OpenAIChatModel(model="qwen-max")
    grader = ContextMemoryGrader(model=model)

    # Complete conversation session
    session = [
        {"role": "user", "content": "I'm allergic to nuts."},
        {"role": "assistant", "content": "I'll remember that."},
        {"role": "user", "content": "Suggest a snack."},
        {"role": "assistant", "content": "Try some fruit or yogurt!"},
        {"role": "user", "content": "And a dessert?"},
        {"role": "assistant", "content": "How about almond cake?"},  # Forgot allergy!
    ]

    # Evaluate each step
    steps = split_session_to_steps(session)
    scores = []

    for i, (history, response) in enumerate(steps):
        result = await grader.aevaluate(history=history, response=response)
        scores.append(result.score)
        print(f"Step {i+1}: {result.score}/5")

    # Aggregate scores
    print(f"\nAverage: {sum(scores)/len(scores):.2f}")
    print(f"Min: {min(scores)}")

asyncio.run(main())
```

**Output:**

```
Step 1: Score=3.0/5
Step 2: Score=5.0/5
Step 3: Score=1.0/5

Average: 3.00
Min: 1.0
```


## Next Steps

- [General Graders](general.md) — Quality assessment (Relevance, Hallucination, Harmfulness)
- [Agent Graders](agent_graders.md) — Agent evaluation (Action, Tool, Memory, Plan)
- [Cookbook Example](../../cookbooks/multi_turn_dialogue/multi_turn_evaluation.py) — Complete evaluation script
