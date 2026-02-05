# -*- coding: utf-8 -*-
"""
Example 2: Built-in ReActAgent + LangChain Tool

Hybrid approach: Use OpenJudge's built-in ReActAgent with LangChain's tool ecosystem.
Best for: Leveraging LangChain's rich tool library without depending on its agent logic.

Dependencies:
    pip install openjudge langchain-tavily

Environment Variables:
    OPENAI_API_KEY: OpenAI-compatible API Key
    TAVILY_API_KEY: Tavily Search API Key

Usage:
    python 02_native_react_langchain_tool.py
"""

import asyncio
import os

from langchain_tavily import TavilySearch as TavilySearchTool

from openjudge.agentic.adapters.langchain import LangChainToolAdapter
from openjudge.graders.agentic_grader import AgenticGrader

# =============================================================================
# Grader Creation
# =============================================================================


def create_grader() -> AgenticGrader:
    """Create a grader using built-in ReActAgent with LangChain tool.

    Returns:
        Configured AgenticGrader instance.
    """
    # 1. Create LangChain tool and wrap it for OpenJudge
    lc_search_tool = TavilySearchTool(max_results=3)
    oj_search_tool = LangChainToolAdapter(lc_search_tool)

    print(f"[DEBUG] LangChain tool name: {oj_search_tool.name}")

    # 2. Define evaluation prompt template
    correctness_template = """
    You are a fact-checking assistant. You must use the available search tool to verify facts.

    **Question:** {query}
    **Response to evaluate:** {response}

    Important: You must call the search tool at least once before scoring.
    Do not rely on your internal knowledge - always verify through search.

    Instructions:
    1. Identify key factual claims in the response.
    2. Use the available search tool to verify each claim.
    3. Score the response (1-5) based on search results.

    Scoring Criteria:
    - 5: All claims verified and accurate.
    - 4: Core facts correct with minor issues.
    - 3: Partially correct, partially wrong.
    - 2: Core facts contradict search results.
    - 1: Completely inaccurate.

    Output in JSON format:
    {{"score": <1-5>, "reason": "<reasoning with search evidence>"}}
    """

    # 3. Create Grader
    grader = AgenticGrader(
        model={
            "model": "qwen3-32b",
            "api_key": os.getenv("OPENAI_API_KEY"),
        },
        tools=[oj_search_tool],
        template=correctness_template,
        name="langchain_tool_correctness_grader",
        max_iterations=5,
    )

    return grader


# =============================================================================
# Main
# =============================================================================


async def main():
    """Run the example."""
    # Check environment variables
    if not os.getenv("TAVILY_API_KEY"):
        print("Error: Please set TAVILY_API_KEY environment variable")
        return

    if not os.getenv("OPENAI_API_KEY"):
        print("Error: Please set OPENAI_API_KEY environment variable")
        return

    # Create Grader
    grader = create_grader()

    print("=" * 60)
    print("Example 2: Built-in ReActAgent + LangChain Tool")
    print("=" * 60)

    # Test case
    query = """Please introduce BYD company, including:
1. Full company name and stock code
2. Main business and core technologies
3. 2024 NEV sales and revenue
4. Global market presence and main models"""

    response = """BYD Company Limited (Stock: A-share 002594, H-share 1211) is a global leader in new energy vehicles.

**Main Business:**
The company focuses on NEVs, power batteries, and rail transit. It masters core technologies including batteries, motors, and electronic controls, being the only company globally that masters both battery and vehicle manufacturing.

**2024 Performance:**
- NEV sales: ~4.26 million units, #1 globally for two consecutive years
- Revenue: ~620 billion RMB
- Net profit: ~35 billion RMB
- Power battery installations: Top 3 globally

**Global Presence & Models:**
- China: Dynasty series (Han, Tang, Song), Ocean series (Seal, Seagull)
- Europe: Germany, France, UK, etc.
- Southeast Asia: Thailand, Singapore, Malaysia
- South America: Factory in Brazil
- Middle East: UAE, Saudi Arabia

BYD has become a global leader in the NEV industry through innovation and globalization."""

    print(f"\nQuery:\n{query}")
    print(f"\nResponse to evaluate:\n{response[:200]}...")
    print("\nEvaluating...")

    result = await grader.aevaluate(query=query, response=response)
    print(f"\nScore: {result.score}")
    print(f"Reason: {result.reason}")
    print(f"Tool calls: {result.metadata.get('tool_calls', 0)}")
    print(f"Iterations: {result.metadata.get('iterations', 0)}")


if __name__ == "__main__":
    asyncio.run(main())
