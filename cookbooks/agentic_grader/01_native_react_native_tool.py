# -*- coding: utf-8 -*-
"""
Example 1: Built-in ReActAgent + Native Tool (Zero Dependencies)

The lightest approach: Use OpenJudge's built-in ReActAgent with custom tools.
Best for: Full control without any third-party agent framework dependencies.

Dependencies:
    pip install openjudge tavily-python

Environment Variables:
    OPENAI_API_KEY: OpenAI-compatible API Key
    TAVILY_API_KEY: Tavily Search API Key

Usage:
    python 01_native_react_native_tool.py
"""

import asyncio
import os

from tavily import TavilyClient

from openjudge.agentic import BaseTool, ToolResult
from openjudge.graders.agentic_grader import AgenticGrader

# =============================================================================
# Tool Definition
# =============================================================================


class TavilySearchTool(BaseTool):
    """Web search tool using Tavily API."""

    schema = {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for information to verify facts.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query string.",
                    }
                },
                "required": ["query"],
            },
        },
    }

    def __init__(self, api_key: str | None = None, max_results: int = 3):
        """Initialize the Tavily search tool.

        Args:
            api_key: Tavily API key. If None, reads from TAVILY_API_KEY env var.
            max_results: Maximum number of search results to return.
        """
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")
        self.max_results = max_results
        self._client = TavilyClient(api_key=self.api_key)

    async def aexecute(self, query: str, **kwargs) -> ToolResult:
        """Execute web search.

        Args:
            query: The search query string.

        Returns:
            ToolResult with search results.
        """
        response = self._client.search(
            query=query,
            search_depth="basic",
            max_results=self.max_results,
        )

        # Format search results
        results = []
        for i, item in enumerate(response.get("results", []), 1):
            title = item.get("title", "No title")
            url = item.get("url", "")
            content = item.get("content", "")[:200]
            results.append(f"[{i}] {title}\nURL: {url}\nContent: {content}...")

        output = "\n\n".join(results) if results else "No results found."
        return ToolResult(success=True, output=output)


# =============================================================================
# Grader Creation
# =============================================================================


def create_grader() -> AgenticGrader:
    """Create a grader using built-in ReActAgent with native tool.

    Returns:
        Configured AgenticGrader instance.
    """
    # 1. Create Tavily search tool
    search_tool = TavilySearchTool()

    # 2. Define evaluation prompt template
    correctness_template = """
    You are a fact-checking assistant. Your task is to verify the correctness of the given response.

    **Question:** {query}
    **Response to evaluate:** {response}

    Instructions:
    1. Use the web_search tool to verify any factual claims in the response.
    2. Compare the search results with the response.
    3. Provide a score from 1-5 based on factual accuracy.

    Scoring Criteria:
    - 5: All factual claims are verified and accurate.
    - 4: Core facts are correct with minor issues.
    - 3: Partially correct with some errors.
    - 2: Core facts contradict search results.
    - 1: Completely inaccurate or fabricated.

    Output your evaluation in JSON format:
    {{"score": <1-5>, "reason": "<your reasoning with search evidence>"}}
    """

    # 3. Create Grader
    grader = AgenticGrader(
        model={
            "model": "qwen3-32b",
            "api_key": os.getenv("OPENAI_API_KEY"),
        },
        tools=[search_tool],
        template=correctness_template,
        name="native_correctness_grader",
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
    print("Example 1: Built-in ReActAgent + Native Tool")
    print("=" * 60)

    # Test case: Complex business question about BYD
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
