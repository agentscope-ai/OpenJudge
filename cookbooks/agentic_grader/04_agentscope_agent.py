# -*- coding: utf-8 -*-
"""
Example 4: AgentScope Agent (Full Delegation)

Full delegation to AgentScope: Use AgentScope's ReActAgent as the reasoning engine.
Best for: Reusing existing AgentScope agents for evaluation tasks.

Dependencies:
    pip install openjudge agentscope tavily-python

Environment Variables:
    OPENAI_API_KEY: OpenAI-compatible API Key
    TAVILY_API_KEY: Tavily Search API Key

Usage:
    python 04_agentscope_agent.py
"""

import asyncio
import os

from agentscope.agent import ReActAgent
from agentscope.formatter import OpenAIChatFormatter
from agentscope.model import OpenAIChatModel
from agentscope.tool import Toolkit, ToolResponse
from tavily import TavilyClient

from openjudge.agentic.adapters.agentscope import AgentScopeAgentAdapter
from openjudge.graders.agentic_grader import AgenticGrader

# =============================================================================
# Tool Definition
# =============================================================================


def create_web_search_tool(toolkit: Toolkit, api_key: str) -> None:
    """Create and register a web search tool with the toolkit.

    Args:
        toolkit: AgentScope Toolkit to register the tool with.
        api_key: Tavily API key.
    """
    client = TavilyClient(api_key=api_key)

    def web_search(query: str) -> ToolResponse:
        """Search the web for information to verify facts.

        Args:
            query: The search query string.

        Returns:
            Search results from the web.
        """
        response = client.search(query=query, search_depth="basic", max_results=3)

        # Format search results
        results = []
        for i, item in enumerate(response.get("results", []), 1):
            title = item.get("title", "No title")
            url = item.get("url", "")
            content = item.get("content", "")[:200]
            results.append(f"[{i}] {title}\nURL: {url}\nContent: {content}...")

        output = "\n\n".join(results) if results else "No results found."
        return ToolResponse(content=output)

    toolkit.register_tool_function(web_search)


# =============================================================================
# Grader Creation
# =============================================================================


def create_grader() -> AgenticGrader:
    """Create a grader using AgentScope Agent.

    Returns:
        Configured AgenticGrader instance.
    """
    # 1. Create AgentScope toolkit with web search tool
    toolkit = Toolkit()
    create_web_search_tool(toolkit, os.getenv("TAVILY_API_KEY"))

    # 2. Create AgentScope Agent
    model = OpenAIChatModel(
        api_key=os.getenv("OPENAI_API_KEY"),
        model_name="qwen3-32b",
    )

    as_agent = ReActAgent(
        name="fact_checker",
        sys_prompt="You are a fact-checking assistant. Verify claims and provide accuracy scores.",
        model=model,
        formatter=OpenAIChatFormatter(),
        toolkit=toolkit,
        max_iters=5,
    )

    # 3. Wrap AgentScope Agent for OpenJudge
    oj_agent = AgentScopeAgentAdapter(as_agent)

    # 4. Define evaluation prompt template
    correctness_template = """
    Evaluate the factual correctness of this response:

    **Question:** {query}
    **Response:** {response}

    Use available tools to verify facts. Output format: {{"score": 0-10, "reason": "..."}}
    """

    # 5. Create Grader
    grader = AgenticGrader(
        agent=oj_agent,
        template=correctness_template,
        name="agentscope_agent_correctness_grader",
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
    print("Example 4: AgentScope Agent (Full Delegation)")
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
