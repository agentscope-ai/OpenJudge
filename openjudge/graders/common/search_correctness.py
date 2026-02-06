# -*- coding: utf-8 -*-
"""
Search-based Correctness Grader.

Evaluates factual accuracy using web search (Tavily API) to verify claims.
Uses ReAct-style autonomous tool calling.
"""

import os
import textwrap
from typing import Optional

from loguru import logger

from openjudge.agentic import BaseTool, ReActAgent, ToolResult
from openjudge.graders.agentic_grader import AgenticGrader
from openjudge.graders.schema import GraderError, GraderMode, GraderScore
from openjudge.models.base_chat_model import BaseChatModel
from openjudge.models.schema.oai.message import ChatMessage
from openjudge.models.schema.prompt_template import LanguageEnum, PromptTemplate

# ============================================================================
# Tavily Search Tool
# ============================================================================


class TavilySearchTool(BaseTool):
    """Web search tool using Tavily API."""

    schema = {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for real-time information to verify facts.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query to find relevant information",
                    },
                    "search_depth": {
                        "type": "string",
                        "enum": ["basic", "advanced"],
                        "description": "Search depth: 'basic' for quick results, 'advanced' for thorough search",
                    },
                },
                "required": ["query"],
            },
        },
    }

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")
        self._client = None

    def _get_client(self):
        if self._client is None:
            from tavily import TavilyClient

            if not self.api_key:
                raise ValueError("Tavily API key not provided.")
            self._client = TavilyClient(self.api_key)
        return self._client

    async def aexecute(self, query: str, search_depth: str = "advanced", **kwargs) -> ToolResult:
        try:
            client = self._get_client()
            response = client.search(query=query, search_depth=search_depth)

            results = response.get("results", [])
            summary_parts = []
            for i, r in enumerate(results[:5]):
                content = r.get("content", "")[:1500]
                summary_parts.append(
                    f"[{i+1}] {r.get('title', '')}\n" f"URL: {r.get('url', '')}\n" f"Content: {content}"
                )

            return ToolResult(
                success=True,
                output="\n\n".join(summary_parts) if summary_parts else "No results found",
                metadata={"query": query, "num_results": len(results)},
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e))


# ============================================================================
# Prompt Template
# ============================================================================

SEARCH_CORRECTNESS_PROMPT_EN = textwrap.dedent(
    """
    You are evaluating the factual accuracy of a response.
    Use web search to verify claims and score accordingly.

    <Query>
    {query}
    </Query>

    <Response>
    {response}
    </Response>

    <Scoring Criteria>
    A factually accurate response should:
    - Contain claims that can be verified through reliable sources.
    - Present information consistent with current, authoritative data.
    - Not fabricate, distort, or misrepresent facts.

    Points should be deducted for:
    - Claims that contradict verified information from search results.
    - Fabricated facts or statistics not supported by any source.
    - Outdated information presented as current.
    - Mixing accurate information with false claims.
    </Scoring Criteria>

    <Guidelines>
    - Use the web_search tool to find relevant information about claims in the response.
    - Compare each claim in the response against search results.
    - Consider the reliability and recency of sources.
    - Note any claims that cannot be verified or are contradicted by sources.
    </Guidelines>

    <Reminder>
    The goal is to evaluate factual accuracy, not writing quality. A well-written response with false claims should
    score low. A simple response with verified facts should score high.
    </Reminder>

    # Output Instructions
    Please use the web_search tool to verify the factual claims in the response, then provide your evaluation in the
    following structured JSON format:
    {{
        "score": <integer from 1 to 5, where 5 means completely accurate and 1 means completely inaccurate>,
        "reason": "<brief explanation of the score, specifically mentioning which claims were verified or contradicted
        by sources>"
    }}

    Scoring Scale:
    - 5: All factual claims in the response are verified by search results, information is accurate and current.
    - 4: Core facts are correct, but there are minor non-critical errors or some information cannot be fully verified.
    - 3: Response contains some correct information, but also has verifiable errors or significant omissions.
    - 2: Core facts in the response contradict search results, or most claims cannot be verified.
    - 1: Response is completely inaccurate, fabricates facts, or contradicts all reliable sources.

    JSON:"""
).strip()

SEARCH_CORRECTNESS_PROMPT_ZH = textwrap.dedent(
    """
    你正在评估一个回答的事实准确性。使用网络搜索验证声明并据此评分。

    <查询>
    {query}
    </查询>

    <回答>
    {response}
    </回答>

    <评分标准>
    事实准确的回答应该：
    - 包含可通过可靠来源验证的声明。
    - 呈现与当前权威数据一致的信息。
    - 不捏造、歪曲或错误陈述事实。

    以下情况应扣分：
    - 声明与搜索结果中的已验证信息相矛盾。
    - 捏造的事实或统计数据，没有任何来源支持。
    - 将过时信息呈现为当前信息。
    - 将准确信息与虚假声明混合。
    </评分标准>

    <指导>
    - 使用 web_search 工具查找与回答中声明相关的信息。
    - 将回答中的每个声明与搜索结果进行比较。
    - 考虑来源的可靠性和时效性。
    - 注意任何无法验证或与来源矛盾的声明。
    </指导>

    <提醒>
    目标是评估事实准确性，而不是写作质量。一个写得很好但包含虚假声明的回答应该得分低。一个简单但事实经过验证的回答应该得分高。
    </提醒>

    # 输出指令
    请使用 web_search 工具验证回答中的事实声明，然后按以下结构化 JSON 格式提供你的评估：
    {{
        "score": <1到5之间的整数，其中5表示完全准确，1表示完全不准确>,
        "reason": "<对所给分数的简要解释，特别提到哪些声明经过验证或与来源矛盾>"
    }}

    评分标尺：
    - 5: 回答中的所有事实声明都经过搜索结果验证，信息准确且时效性强。
    - 4: 回答的核心事实正确，但存在非关键性的小误差或部分信息无法完全验证。
    - 3: 回答包含部分正确信息，但也有可验证的错误或重要遗漏。
    - 2: 回答的核心事实与搜索结果矛盾，或大部分声明无法验证。
    - 1: 回答完全不准确、捏造事实或与所有可靠来源矛盾。

    JSON:"""
).strip()

# Build PromptTemplate
DEFAULT_SEARCH_CORRECTNESS_TEMPLATE = PromptTemplate(
    messages={
        LanguageEnum.EN: [
            ChatMessage(role="user", content=SEARCH_CORRECTNESS_PROMPT_EN),
        ],
        LanguageEnum.ZH: [
            ChatMessage(role="user", content=SEARCH_CORRECTNESS_PROMPT_ZH),
        ],
    },
)


# ============================================================================
# Search Correctness Grader
# ============================================================================


class SearchCorrectnessGrader(AgenticGrader):
    """Grader that uses web search to verify factual accuracy.

    Uses ReAct-style autonomous tool calling - the LLM decides when and
    how to search for information to verify claims in the response.

    Args:
        model: The language model used for evaluation.
        tavily_api_key: Tavily API key for web search. If not provided,
            will try to read from TAVILY_API_KEY environment variable.
        language: Language for prompts (EN or ZH). Defaults to EN.
        max_iterations: Maximum number of ReAct iterations. Defaults to 5.

    Example:
        >>> from openjudge.models import OpenAIChatModel
        >>> model = OpenAIChatModel(model="gpt-4")
        >>> grader = SearchCorrectnessGrader(model=model, tavily_api_key="tvly-...")
        >>> result = await grader.aevaluate(
        ...     query="What is the capital of France?",
        ...     response="The capital of France is Paris."
        ... )
        >>> print(result.score, result.reason)
    """

    def __init__(
        self,
        model: BaseChatModel,
        tavily_api_key: Optional[str] = None,
        language: LanguageEnum = LanguageEnum.EN,
        max_iterations: int = 5,
        **kwargs,
    ):
        search_tool = TavilySearchTool(api_key=tavily_api_key)

        # Build the agent first (unified interface design)
        agent = ReActAgent(
            model=model,
            tools=[search_tool],
            max_iterations=max_iterations,
        )

        super().__init__(
            agent=agent,
            template=DEFAULT_SEARCH_CORRECTNESS_TEMPLATE,
            name="search_correctness",
            mode=GraderMode.POINTWISE,
            description="Factual accuracy verification using web search",
            language=language,
            **kwargs,
        )

    async def _aevaluate(
        self,
        query: str = "",
        response: str = "",
        **kwargs,
    ) -> "GraderScore | GraderError":
        """Evaluate the factual accuracy of a response using web search.

        The grader will use the web_search tool to verify claims in the response,
        then provide a score based on factual accuracy.

        Args:
            query: The original question or query.
            response: The response to evaluate for factual accuracy.
            **kwargs: Additional keyword arguments passed to the template.

        Returns:
            GraderScore: Contains the evaluation result with:
                - name: "search_correctness"
                - score: Integer from 1-5 (5=completely accurate, 1=completely inaccurate)
                - reason: Explanation of the score with verified/contradicted claims
                - metadata: Contains tool_calls count, iterations, total_time, etc.

            GraderError: If evaluation fails, contains error details.

        Example:
            >>> result = await grader.aevaluate(
            ...     query="What are the latest news about Tesla?",
            ...     response="Tesla announced a new partnership with Panasonic for battery production."
            ... )
            >>> if isinstance(result, GraderScore):
            ...     print(f"Score: {result.score}")
            ...     print(f"Reason: {result.reason}")
            ...     print(f"Tool calls: {result.metadata.get('tool_calls', 0)}")
        """

        try:
            # Call parent's _aevaluate directly (AgenticGrader._aevaluate)
            result = await super()._aevaluate(
                query=query,
                response=response,
                **kwargs,
            )
            return GraderScore(
                name=self.name,
                score=result.score,
                reason=result.reason,
                metadata=result.metadata,
            )

        except Exception as e:
            logger.exception(f"Error evaluating search correctness: {e}")
            return GraderError(
                name=self.name,
                error=f"Evaluation error: {str(e)}",
            )


__all__ = ["SearchCorrectnessGrader", "TavilySearchTool"]
