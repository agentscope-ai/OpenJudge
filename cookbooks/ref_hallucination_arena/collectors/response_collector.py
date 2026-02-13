# -*- coding: utf-8 -*-
"""Response collector: collect reference recommendations from target endpoints.

Each endpoint runs with its own concurrency semaphore so that all models are
evaluated simultaneously while respecting per-model rate limits.

Supports two collection modes per endpoint:
  - **Bare mode** (default): Direct LLM call via achat.
  - **Tool-augmented mode** (tool_config.enabled=true): Uses a ReAct agent
    with TavilySearchTool so the LLM can search the web to verify/find real
    papers before recommending them.
"""

import asyncio
from typing import Any, Dict, List, Optional

from loguru import logger

from cookbooks.ref_hallucination_arena.schema import (
    EvaluationConfig,
    OpenAIEndpoint,
    QueryItem,
)
from openjudge.models.openai_chat_model import OpenAIChatModel
from openjudge.models.schema.oai.message import ChatMessage

# ---------------------------------------------------------------------------
# Default system prompt templates
# ---------------------------------------------------------------------------

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

    Each endpoint gets its own asyncio.Semaphore based on its ``max_concurrency``
    config, so all models run in parallel while individually rate-limited.
    """

    def __init__(
        self,
        target_endpoints: Dict[str, OpenAIEndpoint],
        evaluation_config: Optional[EvaluationConfig] = None,
    ):
        self.endpoints = target_endpoints
        self.config = evaluation_config or EvaluationConfig()

        # Per-endpoint resources
        self.models: Dict[str, OpenAIChatModel] = {}
        self.agents: Dict[str, Any] = {}
        self.system_prompts: Dict[str, Optional[str]] = {}
        self._tool_enabled: Dict[str, bool] = {}
        self._semaphores: Dict[str, asyncio.Semaphore] = {}

        # Standard OpenAI SDK params for chat.completions.create().
        # Anything NOT in this set is routed to ``extra_body`` so that
        # provider-specific params (enable_thinking, reasoning, …) are
        # forwarded correctly instead of raising TypeError.
        _STANDARD_PARAMS = {
            "temperature",
            "top_p",
            "n",
            "max_tokens",
            "max_completion_tokens",
            "stop",
            "presence_penalty",
            "frequency_penalty",
            "logit_bias",
            "logprobs",
            "top_logprobs",
            "response_format",
            "seed",
            "tools",
            "tool_choice",
            "user",
            "service_tier",
            "store",
            "metadata",
            "extra_body",
            "extra_headers",
            "extra_query",
            "timeout",
            "stream_options",
            "reasoning_effort",
        }

        for name, endpoint in target_endpoints.items():
            extra_params = dict(endpoint.extra_params or {})
            extra_params.pop("stream", None)

            # Separate provider-specific params into extra_body
            extra_body: Dict[str, Any] = {}
            for key in list(extra_params.keys()):
                if key not in _STANDARD_PARAMS:
                    extra_body[key] = extra_params.pop(key)
            if extra_body:
                extra_params.setdefault("extra_body", {})
                extra_params["extra_body"].update(extra_body)

            self.models[name] = OpenAIChatModel(
                model=endpoint.model,
                api_key=endpoint.api_key,
                base_url=endpoint.base_url,
                stream=False,
                **extra_params,
            )
            self.system_prompts[name] = endpoint.system_prompt
            self._semaphores[name] = asyncio.Semaphore(endpoint.max_concurrency)

            tool_cfg = endpoint.tool_config
            if tool_cfg.enabled:
                self._tool_enabled[name] = True
                self.agents[name] = self._create_tool_agent(self.models[name], tool_cfg)
                logger.info(
                    f"Endpoint '{name}': tool-augmented mode "
                    f"(max_iterations={tool_cfg.max_iterations}, "
                    f"concurrency={endpoint.max_concurrency})"
                )
            else:
                self._tool_enabled[name] = False

            logger.info(f"Endpoint '{name}': max_concurrency={endpoint.max_concurrency}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _create_tool_agent(model: OpenAIChatModel, tool_cfg: Any) -> Any:
        from openjudge.agentic import ReActAgent
        from openjudge.graders.common.search_correctness import TavilySearchTool

        search_tool = TavilySearchTool(api_key=tool_cfg.tavily_api_key)
        return ReActAgent(
            model=model,
            tools=[search_tool],
            max_iterations=tool_cfg.max_iterations,
        )

    def _build_system_prompt(self, endpoint_name: str, query_item: QueryItem) -> str:
        custom = self.system_prompts.get(endpoint_name)
        if custom:
            try:
                return custom.format(num_refs=query_item.num_refs)
            except (KeyError, IndexError):
                return custom

        use_tool = self._tool_enabled.get(endpoint_name, False)
        if use_tool:
            tpl = DEFAULT_TOOL_SYSTEM_PROMPT_ZH if query_item.language == "zh" else DEFAULT_TOOL_SYSTEM_PROMPT_EN
        else:
            tpl = DEFAULT_SYSTEM_PROMPT_ZH if query_item.language == "zh" else DEFAULT_SYSTEM_PROMPT_EN
        return tpl.format(num_refs=query_item.num_refs)

    # ------------------------------------------------------------------
    # Core call logic (unified retry for bare & tool modes)
    # ------------------------------------------------------------------

    async def _call_endpoint(self, endpoint_name: str, query_item: QueryItem) -> Dict[str, Any]:
        """Call a single endpoint with per-endpoint concurrency control and retry."""
        async with self._semaphores[endpoint_name]:
            return await self._call_with_retry(endpoint_name, query_item)

    async def _call_with_retry(self, endpoint_name: str, query_item: QueryItem) -> Dict[str, Any]:
        """Unified retry loop for both bare and tool-augmented modes."""
        use_tool = self._tool_enabled.get(endpoint_name, False)
        timeout = self.config.timeout * (3 if use_tool else 1)
        max_attempts = self.config.retry_times
        last_error = None

        for attempt in range(1, max_attempts + 1):
            try:
                if use_tool:
                    result = await self._do_tool_call(endpoint_name, query_item, timeout)
                else:
                    result = await self._do_bare_call(endpoint_name, query_item, timeout)
                return result
            except asyncio.TimeoutError:
                last_error = "timeout"
                wait = min(10 * attempt, 60)
                logger.warning(f"Timeout {endpoint_name} (attempt {attempt}/{max_attempts})")
            except Exception as e:
                last_error = str(e)
                is_rate_limit = "429" in last_error or "rate" in last_error.lower()
                wait = min((30 if is_rate_limit else 5) * attempt, 180 if is_rate_limit else 60)
                level = "debug" if is_rate_limit else "warning"
                getattr(logger, level)(
                    f"{'Rate limited' if is_rate_limit else 'Error'} on {endpoint_name} "
                    f"(attempt {attempt}/{max_attempts}): {e}"
                )
            await asyncio.sleep(wait)

        logger.warning(f"All {max_attempts} attempts failed for {endpoint_name}: {last_error}")
        return {"endpoint": endpoint_name, "response": None, "success": False, "error": last_error}

    async def _do_bare_call(self, endpoint_name: str, query_item: QueryItem, timeout: float) -> Dict[str, Any]:
        model = self.models[endpoint_name]
        messages = [
            ChatMessage(role="system", content=self._build_system_prompt(endpoint_name, query_item)),
            ChatMessage(role="user", content=query_item.query),
        ]
        response = await asyncio.wait_for(model.achat(messages=messages), timeout=timeout)
        return {"endpoint": endpoint_name, "response": response.content, "success": True}

    @staticmethod
    def _has_bibtex(text: str) -> bool:
        """Check whether *text* contains at least one BibTeX entry."""
        if not text:
            return False
        lower = text.lower()
        return any(
            tag in lower for tag in ("@article", "@inproceedings", "@book", "@misc", "@phdthesis", "@techreport")
        )

    async def _do_tool_call(self, endpoint_name: str, query_item: QueryItem, timeout: float) -> Dict[str, Any]:
        """Run a tool-augmented ReAct call, with a fallback summarisation step.

        When the ReAct agent reaches ``max_iterations`` but the final content
        is just a short "let me search …" stub (i.e. the model was still in
        the middle of tool-calling when the loop ended), we make one additional
        LLM call **without tools** so that the model can synthesise all the
        search results it has gathered so far into proper BibTeX output.
        """
        agent = self.agents[endpoint_name]
        system_prompt = self._build_system_prompt(endpoint_name, query_item)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query_item.query},
        ]
        result = await asyncio.wait_for(agent.arun(messages=messages), timeout=timeout)

        content = result.content or ""
        tool_calls_count = getattr(result, "tool_calls_count", 0)
        iterations = getattr(result, "iterations", 0)
        hit_max = getattr(result, "metadata", {}).get("max_iterations_reached", False)

        # --- Fallback: if the agent exhausted its iterations or the final
        #     content doesn't contain any BibTeX, ask the model one more time
        #     (without tools) to produce the final answer using the context
        #     already accumulated in the conversation. ---
        needs_fallback = not self._has_bibtex(content)
        if needs_fallback:
            reason = "max_iterations reached" if hit_max else "no BibTeX in response"
            logger.info(
                f"Tool-augmented {endpoint_name}: {reason} after "
                f"{tool_calls_count} tool calls in {iterations} iters "
                f"(content length={len(content)}). Running fallback summarisation…"
            )
            content = await self._tool_fallback_summary(
                endpoint_name,
                query_item,
                result,
                timeout,
            )

        logger.info(
            f"Tool-augmented {endpoint_name}: "
            f"{tool_calls_count} tool calls in {iterations} iterations" + (" [with fallback]" if needs_fallback else "")
        )
        return {
            "endpoint": endpoint_name,
            "response": content,
            "success": True,
            "metadata": {
                "tool_augmented": True,
                "tool_calls_count": tool_calls_count,
                "iterations": iterations,
                "used_fallback": needs_fallback,
            },
        }

    async def _tool_fallback_summary(
        self,
        endpoint_name: str,
        query_item: QueryItem,
        agent_result: Any,
        timeout: float,
    ) -> str:
        """Make one final LLM call *without tools* to produce BibTeX output.

        Takes the full conversation history (including all tool-call results)
        and asks the model to produce the final BibTeX answer based on the
        information it has already gathered.
        """
        model = self.models[endpoint_name]

        # Re-use the conversation that the ReAct agent accumulated
        conv_messages = list(getattr(agent_result, "messages", []))

        # Append a user nudge that asks for the final output
        if query_item.language == "zh":
            nudge = (
                "你已经完成了搜索。现在请根据你搜索到的信息，"
                f"以标准BibTeX格式输出{query_item.num_refs}篇真实存在的论文的引用信息。"
                "每条BibTeX后请简述该论文的核心贡献。不要再调用工具，直接给出最终结果。"
            )
        else:
            nudge = (
                "You have finished searching. Now please use the information you gathered "
                f"to output {query_item.num_refs} real papers in standard BibTeX format. "
                "After each BibTeX entry, briefly describe the paper's core contribution. "
                "Do NOT call any more tools — just give the final answer."
            )
        conv_messages.append({"role": "user", "content": nudge})

        try:
            response = await asyncio.wait_for(
                model.achat(messages=conv_messages),  # no tools parameter
                timeout=timeout,
            )
            fallback_content = getattr(response, "content", None) or ""
            logger.info(
                f"Fallback for {endpoint_name}: got {len(fallback_content)} chars, "
                f"has_bibtex={self._has_bibtex(fallback_content)}"
            )
            return fallback_content
        except Exception as e:
            logger.warning(f"Fallback call failed for {endpoint_name}: {e}")
            # Return whatever the agent had — better than nothing
            return agent_result.content or ""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def collect(
        self,
        queries: List[QueryItem],
        on_query_complete: Optional[Any] = None,
        on_single_response: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """Collect responses from all endpoints for all queries.

        All endpoints run concurrently, each bounded by its own semaphore.

        Args:
            queries: List of QueryItem from the user-provided dataset.
            on_query_complete: Optional callback(query_idx, result_dict) called
                as soon as all endpoints for a query are done.
            on_single_response: Optional callback(query_idx, endpoint_name,
                response_text) called as soon as a single endpoint responds
                successfully. Enables streaming verification pipelines.

        Returns:
            List of dicts: {query, discipline, num_refs, responses: {endpoint: text}}
        """
        num_queries = len(queries)
        num_endpoints = len(self.endpoints)
        total_calls = num_queries * num_endpoints
        concurrency_info = ", ".join(f"{n}={self._semaphores[n]._value}" for n in self.endpoints)
        logger.info(
            f"Collecting responses: {num_queries} queries × {num_endpoints} endpoints "
            f"= {total_calls} calls (per-endpoint concurrency: {concurrency_info})"
        )

        # Track per-query completion
        query_results: Dict[int, Dict[str, Any]] = {}
        query_done_counts: Dict[int, int] = {}

        async def _collect_one(query_idx: int, endpoint_name: str) -> Dict[str, Any]:
            result = await self._call_endpoint(endpoint_name, queries[query_idx])
            return {"query_idx": query_idx, "endpoint": endpoint_name, "result": result}

        # Launch ALL tasks at once; semaphores handle per-endpoint throttling
        tasks = [asyncio.ensure_future(_collect_one(i, ep)) for i in range(num_queries) for ep in self.endpoints]

        completed = 0
        for coro in asyncio.as_completed(tasks):
            item = await coro
            completed += 1

            qi, ep, res = item["query_idx"], item["endpoint"], item["result"]

            if qi not in query_results:
                q = queries[qi]
                query_results[qi] = {
                    "query": q.query,
                    "discipline": q.discipline,
                    "num_refs": q.num_refs,
                    "language": q.language,
                    "responses": {},
                }
                query_done_counts[qi] = 0

            query_results[qi]["responses"][ep] = res["response"] if res["success"] else None
            query_done_counts[qi] += 1

            # Fire per-response callback as soon as an endpoint responds
            if res["success"] and on_single_response:
                try:
                    on_single_response(qi, ep, res["response"])
                except Exception as e:
                    logger.warning(f"on_single_response callback error for Q{qi}/{ep}: {e}")

            if query_done_counts[qi] == num_endpoints and on_query_complete:
                try:
                    on_query_complete(qi, query_results[qi])
                except Exception as e:
                    logger.warning(f"on_query_complete callback error for Q{qi}: {e}")

            if completed % 20 == 0 or completed == total_calls:
                logger.info(f"Progress: {completed}/{total_calls} calls completed")

        results = [query_results[i] for i in range(num_queries)]
        success_count = sum(1 for r in results if all(v is not None for v in r["responses"].values()))
        logger.info(f"Collection complete: {success_count}/{num_queries} queries fully successful")
        return results
