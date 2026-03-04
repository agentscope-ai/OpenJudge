# -*- coding: utf-8 -*-
"""
Example 1: Built-in ReActAgent + Native Tool (Zero Dependencies)

Dependencies: pip install openjudge tavily-python
Environment: OPENAI_API_KEY, TAVILY_API_KEY
"""

import asyncio
import os

from tavily import TavilyClient

from openjudge.agentic import BaseTool, ReActAgent, ToolResult
from openjudge.graders.agentic_grader import AgenticGrader


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

    def __init__(self):
        self._client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

    async def aexecute(self, query: str, **kwargs) -> ToolResult:
        response = self._client.search(query=query, max_results=3)
        results = [f"[{i}] {r['title']}: {r['content'][:200]}..." for i, r in enumerate(response.get("results", []), 1)]
        return ToolResult(success=True, output="\n".join(results))


# Create Grader
grader = AgenticGrader(
    agent=ReActAgent(
        model={"model": "qwen3-32b", "api_key": os.getenv("OPENAI_API_KEY")},
        tools=[TavilySearchTool()],
        max_iterations=5,
    ),
    template="""
You are a fact-checking assistant. Your task is to verify the factual accuracy of the given response.

**Question:** {query}
**Response to evaluate:** {response}

Instructions:
1. Identify the key factual claims in the response.
2. Use the web_search tool to verify each claim against reliable sources.
3. Compare the search results with the claims in the response.
4. Provide a score from 1-5 based on factual accuracy.

Scoring Criteria:
- 5: All factual claims are verified and accurate.
- 4: Core facts are correct with minor inaccuracies.
- 3: Partially correct, some claims are wrong.
- 2: Core facts contradict search results.
- 1: Completely inaccurate or fabricated.

Output your evaluation in JSON format:
{{"score": <1-5>, "reason": "<your detailed reasoning with search evidence>"}}
""",
    name="native_correctness_grader",
)

# Evaluate
if __name__ == "__main__":
    query = """Please introduce BYD company, including:
1. Full company name and stock code
2. Main business areas and core technologies
3. 2024 NEV sales volume and global ranking
4. Key markets and popular models"""

    response = """BYD Company Limited (Stock: A-share 002594, H-share 1211) is a global leader in new energy vehicles.

**Main Business:**
The company focuses on NEVs, power batteries, and rail transit. It masters core technologies including batteries, motors, and electronic controls, being the only company globally that masters both battery and vehicle manufacturing.

**2024 Performance:**
- NEV sales: ~4.26 million units, #1 globally for two consecutive years
- Revenue: ~620 billion RMB
- Power battery installations: Top 3 globally

**Global Presence & Models:**
- China: Dynasty series (Han, Tang, Song), Ocean series (Seal, Seagull)
- Europe: Germany, France, UK
- Southeast Asia: Thailand, Singapore, Malaysia
- South America: Factory in Brazil

BYD has become a global leader in the NEV industry through innovation and globalization."""

    result = asyncio.run(grader.aevaluate(query=query, response=response))
    print(f"Score: {result.score}")
    print(f"Reason: {result.reason}")
    print(f"Tool calls: {result.metadata.get('tool_calls', 0)}")
