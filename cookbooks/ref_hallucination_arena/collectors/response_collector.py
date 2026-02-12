# -*- coding: utf-8 -*-
"""Response collector: collect reference recommendations from target endpoints.

Reuses the same pattern as cookbooks.auto_arena.response_collector but adapted
for QueryItem (user-provided dataset) instead of GeneratedQuery.
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


class ResponseCollector:
    """Collect reference recommendation responses from multiple target endpoints.

    For each query, builds an appropriate prompt (with system prompt specifying
    BibTeX output format) and calls all target endpoints concurrently.
    """

    def __init__(
        self,
        target_endpoints: Dict[str, OpenAIEndpoint],
        evaluation_config: Optional[EvaluationConfig] = None,
    ):
        self.endpoints = target_endpoints
        self.config = evaluation_config or EvaluationConfig()

        # Initialize models
        self.models: Dict[str, OpenAIChatModel] = {}
        self.system_prompts: Dict[str, Optional[str]] = {}

        for name, endpoint in target_endpoints.items():
            extra_params = endpoint.extra_params or {}
            extra_params.pop("stream", None)
            self.models[name] = OpenAIChatModel(
                model=endpoint.model,
                api_key=endpoint.api_key,
                base_url=endpoint.base_url,
                stream=False,
                **extra_params,
            )
            self.system_prompts[name] = endpoint.system_prompt

        self.concurrency_manager = ConcurrencyManager()
        self.concurrency_manager.set_max_concurrency(self.config.max_concurrency)

    def _build_system_prompt(
        self,
        endpoint_name: str,
        query_item: QueryItem,
    ) -> str:
        """Build system prompt for a specific endpoint and query.

        If the endpoint has a custom system_prompt, use it (with {num_refs} placeholder).
        Otherwise use the default template based on query language.
        """
        custom = self.system_prompts.get(endpoint_name)
        if custom:
            try:
                return custom.format(num_refs=query_item.num_refs)
            except (KeyError, IndexError):
                return custom

        template = DEFAULT_SYSTEM_PROMPT_ZH if query_item.language == "zh" else DEFAULT_SYSTEM_PROMPT_EN
        return template.format(num_refs=query_item.num_refs)

    async def _call_endpoint(
        self,
        endpoint_name: str,
        query_item: QueryItem,
    ) -> Dict[str, Any]:
        """Call a single endpoint with retry."""
        model = self.models[endpoint_name]
        system_prompt = self._build_system_prompt(endpoint_name, query_item)

        messages = [
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=query_item.query),
        ]

        @retry(
            stop=stop_after_attempt(self.config.retry_times),
            wait=wait_exponential(multiplier=2, min=2, max=60),
            reraise=True,
        )
        async def _call_with_retry():
            return await asyncio.wait_for(
                model.achat(messages=messages),
                timeout=self.config.timeout,
            )

        try:
            response = await _call_with_retry()
            return {
                "endpoint": endpoint_name,
                "response": response.content,
                "success": True,
            }
        except asyncio.TimeoutError:
            logger.warning(f"Timeout calling {endpoint_name} for: {query_item.query[:50]}...")
            return {
                "endpoint": endpoint_name,
                "response": None,
                "success": False,
                "error": "timeout",
            }
        except Exception as e:
            logger.warning(f"Error calling {endpoint_name}: {e}")
            return {
                "endpoint": endpoint_name,
                "response": None,
                "success": False,
                "error": str(e),
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
