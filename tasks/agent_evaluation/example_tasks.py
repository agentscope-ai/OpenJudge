# -*- coding: utf-8 -*-
"""Example tasks for agent evaluation."""

EXAMPLE_TASKS = [
    {
        "name": "basic_math",
        "instruction": "Calculate the sum of 25 and 37, then multiply the result by 4.",
        "expected_output": "The sum of 25 and 37 is 62. Multiplying 62 by 4 gives 248.",
        "category": "math",
        "difficulty": "easy"
    },
    {
        "name": "text_summarization",
        "instruction": "Summarize the following text in 3 sentences: 'Artificial intelligence is transforming industries worldwide. Machine learning algorithms can process vast amounts of data to identify patterns humans might miss. Organizations are increasingly adopting AI solutions to improve efficiency and decision-making.'",
        "expected_output": "AI is transforming industries worldwide. ML algorithms process vast data to identify patterns. Organizations adopt AI to improve efficiency.",
        "category": "text_processing",
        "difficulty": "medium"
    },
    {
        "name": "simple_reasoning",
        "instruction": "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly? Explain your reasoning.",
        "expected_output": "No, we cannot conclude that some roses fade quickly. While all roses are flowers, and some flowers fade quickly, those flowers that fade quickly may not include roses.",
        "category": "logic",
        "difficulty": "medium"
    },
    {
        "name": "fact_question",
        "instruction": "Who was the first person to walk on the moon and when did this happen?",
        "expected_output": "Neil Armstrong was the first person to walk on the moon on July 20, 1969.",
        "category": "knowledge",
        "difficulty": "easy"
    },
    {
        "name": "creative_writing",
        "instruction": "Write a short poem about the changing seasons.",
        "expected_output": "A creative poem about seasons showing spring, summer, fall, and winter.",
        "category": "creativity",
        "difficulty": "medium"
    }
]


def get_task_by_name(name: str):
    """Get a specific task by name."""
    for task in EXAMPLE_TASKS:
        if task["name"] == name:
            return task
    return None


def get_tasks_by_category(category: str):
    """Get all tasks in a specific category."""
    return [task for task in EXAMPLE_TASKS if task["category"] == category]


def get_tasks_by_difficulty(difficulty: str):
    """Get all tasks with a specific difficulty."""
    return [task for task in EXAMPLE_TASKS if task["difficulty"] == difficulty]