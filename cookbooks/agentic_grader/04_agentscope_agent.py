# -*- coding: utf-8 -*-
"""
Example 4: AgentScope Agent (Full Delegation)

Dependencies: pip install openjudge agentscope tavily-python
Environment: OPENAI_API_KEY, TAVILY_API_KEY
"""

import asyncio
import os

from agentscope.agent import ReActAgent as ASReActAgent
from agentscope.formatter import OpenAIChatFormatter
from agentscope.model import OpenAIChatModel
from agentscope.tool import Toolkit, ToolResponse
from tavily import TavilyClient

from cookbooks.agentic_grader.adapters.agentscope import AgentScopeAgentAdapter
from openjudge.graders.agentic_grader import AgenticGrader

# Create AgentScope tool
toolkit = Toolkit()
_tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


@toolkit.register_tool_function
def web_search(query: str) -> ToolResponse:
    """Search the web for information to verify facts.

    Args:
        query: The search query string.

    Returns:
        Search results from the web.
    """
    response = _tavily_client.search(query=query, max_results=3)
    results = [f"[{i}] {r['title']}: {r['content'][:200]}..." for i, r in enumerate(response.get("results", []), 1)]
    return ToolResponse(content="\n".join(results))


# Create AgentScope Agent
as_agent = ASReActAgent(
    name="fact_checker",
    sys_prompt="You are a fact-checking assistant.",
    model=OpenAIChatModel(api_key=os.getenv("OPENAI_API_KEY"), model_name="qwen3-32b"),
    formatter=OpenAIChatFormatter(),
    toolkit=toolkit,
    max_iters=5,
)

# Create Grader
grader = AgenticGrader(
    agent=AgentScopeAgentAdapter(as_agent),
    template="""
You are a fact-checking assistant. Your task is to verify the factual accuracy of the given response.

**Question:** {query}
**Response to evaluate:** {response}

Instructions:
1. Identify the key factual claims in the response.
2. Use the available search tool to verify each claim against reliable sources.
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
    name="agentscope_agent_correctness_grader",
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
