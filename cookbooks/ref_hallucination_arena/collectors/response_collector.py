# -*- coding: utf-8 -*-
"""Response collector: collect reference recommendations from target endpoints.

Reuses the same pattern as cookbooks.auto_arena.response_collector but adapted
for QueryItem (user-provided dataset) instead of GeneratedQuery.

Supports two collection modes per endpoint:
  - **Bare mode** (default): Direct LLM call via achat.
  - **Tool-augmented mode** (tool_config.enabled=true): Uses a ReAct agent
    with TavilySearchTool so the LLM can search the web to verify/find real
    papers before recommending them.
"""

import asyncio
from typing import Any, Dict, List, Optional

from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from cookbooks.ref_hallucination_arena.schema import (
    EvaluationConfig,
    OpenAIEndpoint,
    QueryItem,
)
from openjudge.models.openai_chat_model import OpenAIChatModel
from openjudge.models.schema.oai.message import ChatMessage
from openjudge.utils.concurrency import ConcurrencyManager

# Default system prompt template for reference recommendation
DEFAULT_SYSTEM_PROMPT_ZH = (
    "你是一位学术文献推荐专家。请根据用户的研究主题，推荐{num_refs}篇真实存在的高质量学术论文。"
    "必须以标准BibTeX格式输出每篇论文的引用信息（包含title、author、year、journal/booktitle、doi等字段），"
    "并在每条BibTeX后简述该论文的核心贡献。只推荐你确信真实存在的论文，不要编造。"
)

DEFAULT_SYSTEM_PROMPT_EN = (
    "You are an academic literature recommendation expert. Based on the user's research topic, "
    "recommend {num_refs} real, high-quality academic papers. "
    "Output each paper in standard BibTeX format (including title, author, year, journal/booktitle, doi fields), "
    "and briefly describe each paper's core contribution. Only recommend papers you are confident actually exist."
)

# Tool-augmented system prompts: instruct the model to use web_search to find/verify papers
DEFAULT_TOOL_SYSTEM_PROMPT_ZH = (
    "你是一位学术文献推荐专家。请根据用户的研究主题，推荐{num_refs}篇真实存在的高质量学术论文。\n\n"
    "你可以使用 web_search 工具来搜索和验证论文的真实性。建议你：\n"
    "1. 先用搜索工具查找相关领域的真实论文\n"
    "2. 验证论文的标题、作者、年份等信息的准确性\n"
    "3. 确认论文确实存在后，再以标准BibTeX格式输出\n\n"
    "必须以标准BibTeX格式输出每篇论文的引用信息（包含title、author、year、journal/booktitle、doi等字段），"
    "并在每条BibTeX后简述该论文的核心贡献。只推荐你通过搜索确认真实存在的论文，不要编造。"
)

DEFAULT_TOOL_SYSTEM_PROMPT_EN = (
    "You are an academic literature recommendation expert. Based on the user's research topic, "
    "recommend {num_refs} real, high-quality academic papers.\n\n"
    "You have access to a web_search tool to search and verify papers. You should:\n"
    "1. Use the search tool to find real papers in the relevant field\n"
    "2. Verify the accuracy of paper titles, authors, years, and other metadata\n"
    "3. Only output papers after confirming they actually exist\n\n"
    "Output each paper in standard BibTeX format (including title, author, year, journal/booktitle, doi fields), "
    "and briefly describe each paper's core contribution. Only recommend papers you have verified to be real."
)


class ResponseCollector:
    """Collect reference recommendation responses from multiple target endpoints.

    For each query, builds an appropriate prompt (with system prompt specifying
    BibTeX output format) and calls all target endpoints concurrently.

    Supports two modes per endpoint:
      - **Bare mode** (default): Direct LLM call via model.achat().
      - **Tool-augmented mode** (tool_config.enabled=true): Uses a ReAct agent
        with TavilySearchTool. The model can autonomously search the web to
        find and verify real papers before recommending them.
    """

    def __init__(
        self,
        target_endpoints: Dict[str, OpenAIEndpoint],
        evaluation_config: Optional[EvaluationConfig] = None,
    ):
        self.endpoints = target_endpoints
        self.config = evaluation_config or EvaluationConfig()

        # Initialize models and optional ReAct agents
        self.models: Dict[str, OpenAIChatModel] = {}
        self.agents: Dict[str, Any] = {}  # endpoint_name -> ReActAgent (only for tool-augmented)
        self.system_prompts: Dict[str, Optional[str]] = {}
        self._tool_enabled: Dict[str, bool] = {}  # track which endpoints use tools

        for name, endpoint in target_endpoints.items():
            extra_params = endpoint.extra_params or {}
            extra_params.pop("stream", None)
            model = OpenAIChatModel(
                model=endpoint.model,
                api_key=endpoint.api_key,
                base_url=endpoint.base_url,
                stream=False,
                **extra_params,
            )
            self.models[name] = model
            self.system_prompts[name] = endpoint.system_prompt

            # Initialize ReAct agent if tool-augmented mode is enabled
            tool_cfg = endpoint.tool_config
            if tool_cfg.enabled:
                self._tool_enabled[name] = True
                agent = self._create_tool_agent(model, tool_cfg)
                self.agents[name] = agent
                logger.info(
                    f"Endpoint '{name}': tool-augmented mode enabled "
                    f"(max_iterations={tool_cfg.max_iterations})"
                )
            else:
                self._tool_enabled[name] = False

        self.concurrency_manager = ConcurrencyManager()
        self.concurrency_manager.set_max_concurrency(self.config.max_concurrency)

    @staticmethod
    def _create_tool_agent(model: OpenAIChatModel, tool_cfg: Any) -> Any:
        """Create a ReAct agent with TavilySearchTool for tool-augmented mode.

        Args:
            model: The OpenAI chat model instance.
            tool_cfg: ToolConfig with tavily_api_key and max_iterations.

        Returns:
            Configured ReActAgent instance.
        """
        from openjudge.agentic import ReActAgent
        from openjudge.graders.common.search_correctness import TavilySearchTool

        search_tool = TavilySearchTool(api_key=tool_cfg.tavily_api_key)
        return ReActAgent(
            model=model,
            tools=[search_tool],
            max_iterations=tool_cfg.max_iterations,
        )

    def _build_system_prompt(
        self,
        endpoint_name: str,
        query_item: QueryItem,
    ) -> str:
        """Build system prompt for a specific endpoint and query.

        If the endpoint has a custom system_prompt, use it (with {num_refs} placeholder).
        Otherwise use the default template based on query language and tool mode.
        """
        custom = self.system_prompts.get(endpoint_name)
        if custom:
            try:
                return custom.format(num_refs=query_item.num_refs)
            except (KeyError, IndexError):
                return custom

        # Select template based on tool mode and language
        use_tool = self._tool_enabled.get(endpoint_name, False)
        if use_tool:
            template = (
                DEFAULT_TOOL_SYSTEM_PROMPT_ZH
                if query_item.language == "zh"
                else DEFAULT_TOOL_SYSTEM_PROMPT_EN
            )
        else:
            template = (
                DEFAULT_SYSTEM_PROMPT_ZH
                if query_item.language == "zh"
                else DEFAULT_SYSTEM_PROMPT_EN
            )
        return template.format(num_refs=query_item.num_refs)

    async def _call_endpoint(
        self,
        endpoint_name: str,
        query_item: QueryItem,
    ) -> Dict[str, Any]:
        """Call a single endpoint with retry.

        Uses bare achat for normal endpoints, or ReActAgent for tool-augmented
        endpoints. For 429 rate-limit errors, uses longer waits between retries.
        """
        use_tool = self._tool_enabled.get(endpoint_name, False)

        if use_tool:
            return await self._call_endpoint_with_tools(endpoint_name, query_item)
        else:
            return await self._call_endpoint_bare(endpoint_name, query_item)

    async def _call_endpoint_bare(
        self,
        endpoint_name: str,
        query_item: QueryItem,
    ) -> Dict[str, Any]:
        """Call endpoint in bare mode (direct achat, no tools)."""
        model = self.models[endpoint_name]
        system_prompt = self._build_system_prompt(endpoint_name, query_item)

        messages = [
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=query_item.query),
        ]

        max_attempts = self.config.retry_times
        last_error = None

        for attempt in range(1, max_attempts + 1):
            try:
                response = await asyncio.wait_for(
                    model.achat(messages=messages),
                    timeout=self.config.timeout,
                )
                return {
                    "endpoint": endpoint_name,
                    "response": response.content,
                    "success": True,
                }
            except asyncio.TimeoutError:
                logger.warning(f"Timeout calling {endpoint_name} (attempt {attempt}/{max_attempts})")
                last_error = "timeout"
                wait_time = min(10 * attempt, 60)
                await asyncio.sleep(wait_time)
            except Exception as e:
                last_error = str(e)
                is_rate_limit = "429" in str(e) or "rate" in str(e).lower()
                if is_rate_limit:
                    wait_time = min(30 * attempt, 180)
                    logger.debug(
                        f"Rate limited on {endpoint_name} (attempt {attempt}/{max_attempts}), "
                        f"waiting {wait_time}s..."
                    )
                else:
                    wait_time = min(5 * attempt, 60)
                    logger.warning(
                        f"Error calling {endpoint_name} (attempt {attempt}/{max_attempts}): {e}"
                    )
                await asyncio.sleep(wait_time)

        logger.warning(f"All {max_attempts} attempts failed for {endpoint_name}: {last_error}")
        return {
            "endpoint": endpoint_name,
            "response": None,
            "success": False,
            "error": last_error,
        }

    async def _call_endpoint_with_tools(
        self,
        endpoint_name: str,
        query_item: QueryItem,
    ) -> Dict[str, Any]:
        """Call endpoint in tool-augmented mode using ReAct agent.

        The ReAct agent drives a loop where the LLM can autonomously call
        web_search to find and verify real papers before producing its final
        BibTeX output.
        """
        agent = self.agents[endpoint_name]
        system_prompt = self._build_system_prompt(endpoint_name, query_item)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query_item.query},
        ]

        max_attempts = self.config.retry_times
        last_error = None

        # Tool-augmented mode needs longer timeout (multiple search rounds)
        tool_timeout = self.config.timeout * 3

        for attempt in range(1, max_attempts + 1):
            try:
                agent_result = await asyncio.wait_for(
                    agent.arun(messages=messages),
                    timeout=tool_timeout,
                )
                tool_calls_count = getattr(agent_result, "tool_calls_count", 0)
                iterations = getattr(agent_result, "iterations", 0)
                logger.info(
                    f"Tool-augmented {endpoint_name}: "
                    f"{tool_calls_count} tool calls in {iterations} iterations"
                )
                return {
                    "endpoint": endpoint_name,
                    "response": agent_result.content,
                    "success": True,
                    "metadata": {
                        "tool_augmented": True,
                        "tool_calls_count": tool_calls_count,
                        "iterations": iterations,
                    },
                }
            except asyncio.TimeoutError:
                logger.warning(
                    f"Timeout calling tool-augmented {endpoint_name} "
                    f"(attempt {attempt}/{max_attempts}, timeout={tool_timeout}s)"
                )
                last_error = "timeout"
                wait_time = min(10 * attempt, 60)
                await asyncio.sleep(wait_time)
            except Exception as e:
                last_error = str(e)
                is_rate_limit = "429" in str(e) or "rate" in str(e).lower()
                if is_rate_limit:
                    wait_time = min(30 * attempt, 180)
                    logger.debug(
                        f"Rate limited on tool-augmented {endpoint_name} "
                        f"(attempt {attempt}/{max_attempts}), waiting {wait_time}s..."
                    )
                else:
                    wait_time = min(5 * attempt, 60)
                    logger.warning(
                        f"Error calling tool-augmented {endpoint_name} "
                        f"(attempt {attempt}/{max_attempts}): {e}"
                    )
                await asyncio.sleep(wait_time)

        logger.warning(f"All {max_attempts} attempts failed for tool-augmented {endpoint_name}: {last_error}")
        return {
            "endpoint": endpoint_name,
            "response": None,
            "success": False,
            "error": last_error,
        }

    async def collect(
        self,
        queries: List[QueryItem],
        on_query_complete: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """Collect responses from all endpoints for all queries.

        Args:
            queries: List of QueryItem from the user-provided dataset.
            on_query_complete: Optional callback(query_idx, result_dict) called
                as soon as all endpoints for a query are done.  This enables
                the caller to persist results incrementally.

        Returns:
            List of dicts: {query, discipline, num_refs, responses: {endpoint: text}}
        """
        total_calls = len(queries) * len(self.endpoints)
        num_endpoints = len(self.endpoints)
        logger.info(
            f"Collecting responses for {len(queries)} queries × "
            f"{num_endpoints} endpoints = {total_calls} calls "
            f"(max_concurrency={self.config.max_concurrency})"
        )

        # Track per-query endpoint results so we can fire callback when a
        # query's ALL endpoints are done.
        query_ep_results: Dict[int, Dict[str, Any]] = {}
        query_ep_counts: Dict[int, int] = {}

        async def _collect_one(query_idx: int, endpoint_name: str) -> Dict[str, Any]:
            query_item = queries[query_idx]
            result = await self._call_endpoint(endpoint_name, query_item)
            return {
                "query_idx": query_idx,
                "endpoint": endpoint_name,
                "result": result,
            }

        tasks = [
            self.concurrency_manager.run_with_concurrency_control(_collect_one(i, ep_name))
            for i in range(len(queries))
            for ep_name in self.endpoints
        ]

        completed = 0
        for coro in asyncio.as_completed(tasks):
            item = await coro
            completed += 1

            qi = item["query_idx"]
            ep = item["endpoint"]
            res = item["result"]

            if qi not in query_ep_results:
                q = queries[qi]
                query_ep_results[qi] = {
                    "query": q.query,
                    "discipline": q.discipline,
                    "num_refs": q.num_refs,
                    "language": q.language,
                    "responses": {},
                }
                query_ep_counts[qi] = 0

            if res["success"]:
                query_ep_results[qi]["responses"][ep] = res["response"]
            else:
                query_ep_results[qi]["responses"][ep] = None

            query_ep_counts[qi] += 1

            # When all endpoints for this query are done, fire callback
            if query_ep_counts[qi] == num_endpoints and on_query_complete:
                try:
                    on_query_complete(qi, query_ep_results[qi])
                except Exception as e:
                    logger.warning(f"on_query_complete callback error for Q{qi}: {e}")

            if completed % 10 == 0 or completed == total_calls:
                logger.info(f"Progress: {completed}/{total_calls} calls completed")

        results = [query_ep_results[i] for i in range(len(queries))]

        success_count = sum(1 for r in results if all(v is not None for v in r["responses"].values()))
        logger.info(f"Collection complete: {success_count}/{len(results)} queries fully successful")

        return results
