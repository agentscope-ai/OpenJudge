# -*- coding: utf-8 -*-
"""Response collector for zero-shot evaluation."""

import asyncio
from typing import Any, Dict, List, Optional

from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from cookbooks.zero_shot_evaluation.schema import (
    EvaluationConfig,
    GeneratedQuery,
    OpenAIEndpoint,
)
from openjudge.models.openai_chat_model import OpenAIChatModel
from openjudge.models.schema.oai.message import ChatMessage
from openjudge.utils.concurrency import ConcurrencyManager


class ResponseCollector:
    """Collect responses from multiple target endpoints."""

    def __init__(
        self,
        target_endpoints: Dict[str, OpenAIEndpoint],
        evaluation_config: Optional[EvaluationConfig] = None,
    ):
        """Initialize ResponseCollector.

        Args:
            target_endpoints: Dictionary of endpoint name to configuration
            evaluation_config: Evaluation configuration
        """
        self.endpoints = target_endpoints
        self.config = evaluation_config or EvaluationConfig()

        # Initialize models for each endpoint (force stream=False)
        self.models: Dict[str, OpenAIChatModel] = {}
        self.system_prompts: Dict[str, Optional[str]] = {}

        for name, endpoint in target_endpoints.items():
            extra_params = endpoint.extra_params or {}
            # Ensure stream is disabled to avoid async generator issues
            extra_params.pop("stream", None)
            self.models[name] = OpenAIChatModel(
                model=endpoint.model,
                api_key=endpoint.api_key,
                base_url=endpoint.base_url,
                stream=False,
                **extra_params,
            )
            self.system_prompts[name] = endpoint.system_prompt

        # Setup concurrency manager (use singleton's set method)
        self.concurrency_manager = ConcurrencyManager()
        self.concurrency_manager.set_max_concurrency(self.config.max_concurrency)

    async def _call_endpoint(
        self,
        endpoint_name: str,
        query: str,
    ) -> Dict[str, Any]:
        """Call a single endpoint with a query (with retry).

        Args:
            endpoint_name: Name of the endpoint
            query: Query text

        Returns:
            Dictionary with response or error
        """
        model = self.models[endpoint_name]
        system_prompt = self.system_prompts[endpoint_name]

        messages: List[ChatMessage] = []
        if system_prompt:
            messages.append(ChatMessage(role="system", content=system_prompt))
        messages.append(ChatMessage(role="user", content=query))

        # Create retry decorator with configured retry_times
        @retry(
            stop=stop_after_attempt(self.config.retry_times),
            wait=wait_exponential(multiplier=1, min=1, max=10),
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
            logger.warning(f"Timeout calling {endpoint_name} for query: {query[:50]}...")
            return {
                "endpoint": endpoint_name,
                "response": None,
                "success": False,
                "error": "timeout",
            }
        except Exception as e:
            logger.warning(f"Error calling {endpoint_name} after {self.config.retry_times} retries: {e}")
            return {
                "endpoint": endpoint_name,
                "response": None,
                "success": False,
                "error": str(e),
            }

    async def collect_single(self, query: str) -> Dict[str, Any]:
        """Collect responses from all endpoints for a single query.

        Args:
            query: Query text

        Returns:
            Dictionary mapping endpoint names to responses
        """
        tasks = [
            self.concurrency_manager.run_with_concurrency_control(
                self._call_endpoint(name, query),
            )
            for name in self.endpoints
        ]

        results = await asyncio.gather(*tasks)

        responses = {}
        for result in results:
            endpoint_name = result["endpoint"]
            if result["success"]:
                responses[endpoint_name] = result["response"]
            else:
                responses[endpoint_name] = None
                logger.debug(f"Failed response from {endpoint_name}: {result.get('error')}")

        return responses

    async def collect(
        self,
        queries: List[GeneratedQuery],
    ) -> List[Dict[str, Any]]:
        """Collect responses from all endpoints for all queries (fully parallel).

        Args:
            queries: List of GeneratedQuery objects

        Returns:
            List of dictionaries, each containing query and responses
        """
        total_calls = len(queries) * len(self.endpoints)
        logger.info(
            f"Collecting responses for {len(queries)} queries from {len(self.endpoints)} endpoints "
            f"({total_calls} total calls, max_concurrency={self.config.max_concurrency})"
        )

        # Create all (query_idx, endpoint_name) tasks
        async def _collect_one(query_idx: int, endpoint_name: str) -> Dict[str, Any]:
            query_obj = queries[query_idx]
            result = await self._call_endpoint(endpoint_name, query_obj.query)
            return {
                "query_idx": query_idx,
                "endpoint": endpoint_name,
                "result": result,
            }

        # Launch all tasks with concurrency control
        tasks = [
            self.concurrency_manager.run_with_concurrency_control(_collect_one(i, ep_name))
            for i in range(len(queries))
            for ep_name in self.endpoints
        ]

        # Progress tracking
        completed = 0
        all_results = []

        for coro in asyncio.as_completed(tasks):
            result = await coro
            all_results.append(result)
            completed += 1
            if completed % 10 == 0 or completed == total_calls:
                logger.info(f"Progress: {completed}/{total_calls} calls completed")

        # Organize results by query
        results_by_query: Dict[int, Dict[str, Any]] = {}
        for item in all_results:
            query_idx = item["query_idx"]
            endpoint = item["endpoint"]
            result = item["result"]

            if query_idx not in results_by_query:
                query_obj = queries[query_idx]
                results_by_query[query_idx] = {
                    "query": query_obj.query,
                    "category": query_obj.category,
                    "difficulty": query_obj.difficulty,
                    "responses": {},
                }

            if result["success"]:
                results_by_query[query_idx]["responses"][endpoint] = result["response"]
            else:
                results_by_query[query_idx]["responses"][endpoint] = None
                logger.debug(f"Failed response from {endpoint}: {result.get('error')}")

        # Convert to ordered list
        results = [results_by_query[i] for i in range(len(queries))]

        # Log summary
        success_count = sum(1 for r in results if all(v is not None for v in r["responses"].values()))
        logger.info(f"Collected responses: {success_count}/{len(results)} queries fully successful")

        return results
