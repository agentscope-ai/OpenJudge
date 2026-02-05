# -*- coding: utf-8 -*-
"""
Example 3: LangChain Agent (Full Delegation)

Full delegation to LangChain: Use LangChain's create_agent as the reasoning engine.
Best for: Reusing existing LangChain agents for evaluation tasks.

Dependencies:
    pip install openjudge langchain langchain-openai langchain-tavily

Environment Variables:
    OPENAI_API_KEY: OpenAI-compatible API Key
    TAVILY_API_KEY: Tavily Search API Key

Usage:
    python 03_langchain_agent.py
"""

import asyncio
import os

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch as TavilySearchTool

from openjudge.agentic.adapters.langchain import LangChainAgentAdapter
from openjudge.graders.agentic_grader import AgenticGrader

# =============================================================================
# Grader Creation
# =============================================================================


def create_grader() -> AgenticGrader:
    """Create a grader using LangChain Agent.

    Returns:
        Configured AgenticGrader instance.
    """
    # 1. Create LangChain tools
    tools = [TavilySearchTool(max_results=3)]

    # 2. Create LangChain ReAct Agent
    # Note: qwen models need to disable thinking mode
    llm = ChatOpenAI(
        model="qwen3-32b",
        api_key=os.getenv("OPENAI_API_KEY"),
        extra_body={"enable_thinking": False},
    )

    lc_agent = create_agent(llm, tools)

    # 3. Wrap LangChain Agent for OpenJudge
    oj_agent = LangChainAgentAdapter(lc_agent)

    # 4. Define evaluation prompt template
    correctness_template = """
    Please evaluate the factual correctness of the following response:

    **Question:** {query}
    **Response:** {response}

    Use web search to verify facts. Output in JSON format: {{"score": 1-5, "reason": "..."}}
    """

    # 5. Create Grader
    grader = AgenticGrader(
        agent=oj_agent,
        template=correctness_template,
        name="langchain_agent_correctness_grader",
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
    print("Example 3: LangChain Agent (Full Delegation)")
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
