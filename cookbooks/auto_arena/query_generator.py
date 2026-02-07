# -*- coding: utf-8 -*-
"""Query generator for Auto Arena evaluation with advanced optimization strategies.

Features:
- Iterative generation with deduplication
- Evol-Instruct style complexity evolution
- Async parallel batch generation
"""

import asyncio
import hashlib
from collections import defaultdict
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Set, Tuple

from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from cookbooks.auto_arena.schema import (
    GeneratedQuery,
    OpenAIEndpoint,
    QueryGenerationConfig,
    QueryGenerationOutput,
    TaskConfig,
)
from openjudge.models.openai_chat_model import OpenAIChatModel
from openjudge.models.schema.oai.message import ChatMessage
from openjudge.models.schema.prompt_template import PromptTemplate

# =============================================================================
# Prompt Templates
# =============================================================================

QUERY_GENERATION_PROMPT = """# Task
Based on the task description, generate diverse and representative test queries.

## Task Description
{task_description}

## Scenario
{scenario}

## Seed Queries (for reference)
{seed_queries}

## Category Distribution
{categories}

## Already Generated Queries (AVOID similar ones)
{existing_queries}

## Diversity Requirements
- Cover different query lengths (short: <20 words, medium: 20-50 words, long: >50 words)
- Include various difficulty levels (easy, medium, hard)
- Vary question types (factual, analytical, creative, edge cases)
- Include both common scenarios and edge cases
- AVOID semantically similar or redundant queries to existing ones

## Anti-patterns to AVOID
- Don't generate queries too similar to existing ones listed above
- Don't generate overly generic queries like "Tell me about X"
- Don't repeat the same query structure with minor word changes
- Don't use template-like patterns repeatedly

## Requirements
- Generate exactly {num_queries} test queries
- Each query should be independent and self-contained
- Batch ID: {batch_id} (use this to vary your generation strategy)

## Output Format
Return a JSON object with:
- queries: list of objects, each with "query" (required), "category" (optional), "difficulty" (optional)
- reason: brief explanation of generation strategy

Example:
{{
    "queries": [
        {{"query": "How does X handle Y in scenario Z?", "category": "technical", "difficulty": "medium"}},
        {{"query": "What happens when...", "category": "edge_case", "difficulty": "hard"}}
    ],
    "reason": "Generated queries covering different aspects..."
}}
"""

EVOLUTION_PROMPT = """# Task
Evolve the given query into more complex versions using the specified strategy.

## Original Query
{original_query}

## Evolution Strategy: {strategy}

### Strategy Descriptions:
- constraints: Add specific constraints (time, scope, conditions, limitations)
- reasoning: Require multi-step reasoning or comparison
- edge_cases: Add edge cases, exceptions, or unusual conditions
- combination: Combine with related concepts or cross-domain knowledge

## Requirements
- Generate {num_variations} evolved versions
- Each version should be more challenging than the original
- Maintain the core intent while increasing complexity
- Evolved queries should be natural and realistic

## Output Format
Return a JSON object with:
- evolved_queries: list of objects with "query", "difficulty", "evolution_type"
- reasoning: explanation of how complexity was increased

Example:
{{
    "evolved_queries": [
        {{"query": "...", "difficulty": "hard", "evolution_type": "constraints"}},
        {{"query": "...", "difficulty": "hard", "evolution_type": "reasoning"}}
    ],
    "reasoning": "Added time constraints and multi-step reasoning..."
}}
"""

QUERY_GENERATION_TEMPLATE = PromptTemplate(
    messages=[
        ChatMessage(
            role="system",
            content="You are an expert at generating diverse and representative test queries for AI evaluation. "
            "You excel at creating queries that cover various difficulty levels, categories, and edge cases. "
            "You MUST avoid generating duplicate or semantically similar queries.",
        ),
        ChatMessage(role="user", content=QUERY_GENERATION_PROMPT),
    ],
)

EVOLUTION_TEMPLATE = PromptTemplate(
    messages=[
        ChatMessage(
            role="system",
            content="You are an expert at evolving simple queries into more complex, challenging versions. "
            "You apply the Evol-Instruct methodology to increase query complexity while maintaining naturalness.",
        ),
        ChatMessage(role="user", content=EVOLUTION_PROMPT),
    ],
)


# =============================================================================
# Evolution Output Schema
# =============================================================================


from pydantic import BaseModel, Field


class EvolvedQuery(BaseModel):
    """Single evolved query."""

    query: str = Field(..., description="The evolved query text")
    difficulty: str = Field(default="hard", description="Difficulty level")
    evolution_type: str = Field(default="", description="Type of evolution applied")


class EvolutionOutput(BaseModel):
    """Output schema for query evolution."""

    evolved_queries: List[EvolvedQuery] = Field(..., description="List of evolved queries")
    reasoning: str = Field(default="", description="Evolution reasoning")


# =============================================================================
# Query Deduplicator
# =============================================================================


class QueryDeduplicator:
    """Handles query deduplication using multiple strategies."""

    def __init__(self, max_similarity: float = 0.85):
        self.max_similarity = max_similarity
        self._seen_hashes: Set[str] = set()
        self._seen_queries: List[str] = []

    def _normalize(self, text: str) -> str:
        """Normalize text for comparison."""
        return " ".join(text.lower().strip().split())

    def _hash(self, text: str) -> str:
        """Create hash for exact deduplication."""
        normalized = self._normalize(text)
        return hashlib.md5(normalized.encode()).hexdigest()

    def _similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity ratio between two texts."""
        return SequenceMatcher(None, self._normalize(text1), self._normalize(text2)).ratio()

    def is_duplicate(self, query: str) -> bool:
        """Check if query is a duplicate."""
        query_hash = self._hash(query)

        # Exact duplicate check
        if query_hash in self._seen_hashes:
            return True

        # Semantic similarity check (against recent queries for efficiency)
        check_against = self._seen_queries[-100:] if len(self._seen_queries) > 100 else self._seen_queries
        for seen in check_against:
            if self._similarity(query, seen) > self.max_similarity:
                return True

        return False

    def add(self, query: str) -> bool:
        """Add query if not duplicate. Returns True if added."""
        if self.is_duplicate(query):
            return False

        self._seen_hashes.add(self._hash(query))
        self._seen_queries.append(query)
        return True

    def get_existing_summary(self, max_items: int = 10) -> str:
        """Get summary of existing queries for prompt context."""
        if not self._seen_queries:
            return "None yet"

        # Sample from existing queries
        sample = self._seen_queries[:max_items]
        return "\n".join(f"- {q[:100]}..." if len(q) > 100 else f"- {q}" for q in sample)


# =============================================================================
# Query Validator
# =============================================================================


class QueryValidator:
    """Validate generated queries for quality."""

    MIN_LENGTH = 5
    MAX_LENGTH = 1000

    @classmethod
    def validate(cls, query: GeneratedQuery) -> Tuple[bool, str]:
        """Validate a single query. Returns (is_valid, reason)."""
        text = query.query.strip()

        if len(text) < cls.MIN_LENGTH:
            return False, f"Too short: {len(text)} chars"

        if len(text) > cls.MAX_LENGTH:
            return False, f"Too long: {len(text)} chars"

        # Check for placeholder patterns
        placeholders = ["[", "]", "{", "}", "...", "___", "XXX"]
        for p in placeholders:
            if p in text and text.count(p) > 2:
                return False, f"Contains placeholder pattern: {p}"

        return True, "OK"


# =============================================================================
# Query Generator
# =============================================================================


class QueryGenerator:
    """Generate test queries with advanced optimization strategies.

    Features:
    - Iterative batch generation with deduplication
    - Evol-Instruct style complexity evolution
    - Async parallel generation for efficiency
    - Configurable endpoint for query generation
    """

    def __init__(
        self,
        judge_endpoint: OpenAIEndpoint,
        task_config: TaskConfig,
        query_config: Optional[QueryGenerationConfig] = None,
    ):
        """Initialize QueryGenerator.

        Args:
            judge_endpoint: OpenAI-compatible endpoint (fallback if query_config.endpoint not set)
            task_config: Task configuration
            query_config: Query generation configuration (including optional custom endpoint)
        """
        self.task_config = task_config
        self.query_config = query_config or QueryGenerationConfig()

        # Initialize deduplicator
        self.deduplicator = QueryDeduplicator(max_similarity=self.query_config.max_similarity)

        # Determine which endpoint to use: custom endpoint in query_config, or fallback to judge_endpoint
        if self.query_config.endpoint:
            # Use custom endpoint specified in query_generation config
            endpoint = self.query_config.endpoint
            extra_params = endpoint.extra_params or {}
            logger.info(f"Using custom query generation endpoint: {endpoint.model} @ {endpoint.base_url}")
        else:
            # Fallback to judge_endpoint
            endpoint = judge_endpoint
            extra_params = endpoint.extra_params or {}
            logger.info(f"Using judge endpoint for query generation: {endpoint.model} @ {endpoint.base_url}")

        extra_params = dict(extra_params)  # Make a copy to avoid modifying original
        # Remove params that we'll set explicitly to avoid conflicts
        extra_params.pop("stream", None)
        extra_params.pop("temperature", None)
        extra_params.pop("top_p", None)

        self.model = OpenAIChatModel(
            model=endpoint.model,
            api_key=endpoint.api_key,
            base_url=endpoint.base_url,
            stream=False,
            temperature=self.query_config.temperature,
            top_p=self.query_config.top_p,
            **extra_params,
        )

    # =========================================================================
    # Main Generation Entry Point
    # =========================================================================

    async def generate(self, max_retries: int = 5) -> List[GeneratedQuery]:
        """Generate test queries with all optimization strategies.

        Pipeline:
        1. Parallel batch generation (with retry until target count is reached)
        2. Deduplication
        3. Optional complexity evolution
        4. Final validation and filtering

        Args:
            max_retries: Maximum number of retry rounds if not enough queries generated

        Returns:
            List of GeneratedQuery objects
        """
        target_count = self.query_config.num_queries
        logger.info(
            f"Starting query generation: target={target_count}, "
            f"queries_per_call={self.query_config.queries_per_call}, "
            f"parallel_batches={self.query_config.num_parallel_batches}, "
            f"evolution={'enabled' if self.query_config.enable_evolution else 'disabled'}"
        )

        # Step 1: Parallel batch generation with retry until target count
        base_queries: List[GeneratedQuery] = []
        retry_round = 0
        consecutive_failures = 0
        max_consecutive_failures = 3  # Stop after 3 consecutive complete failures

        while len(base_queries) < target_count and retry_round <= max_retries:
            if retry_round > 0:
                remaining = target_count - len(base_queries)
                logger.info(f"Retry round {retry_round}: need {remaining} more queries")

            new_queries = await self._parallel_generate()

            # Deduplicate against existing queries
            added_count = 0
            for q in new_queries:
                if not self._is_duplicate(q, base_queries):
                    base_queries.append(q)
                    added_count += 1

            logger.info(f"After round {retry_round}: {len(base_queries)} queries collected (+{added_count} new)")

            if len(new_queries) == 0:
                consecutive_failures += 1
                logger.warning(
                    f"Round {retry_round} produced 0 queries (consecutive failures: {consecutive_failures}/{max_consecutive_failures})"
                )
                if consecutive_failures >= max_consecutive_failures:
                    logger.error(f"Stopping after {max_consecutive_failures} consecutive complete failures")
                    break
            else:
                consecutive_failures = 0  # Reset on any success

            retry_round += 1

        logger.info(f"Base generation complete: {len(base_queries)} queries")

        # Step 2: Optional complexity evolution
        if self.query_config.enable_evolution and self.query_config.evolution_rounds > 0:
            evolved_queries = await self._evolve_queries(base_queries)
            base_queries.extend(evolved_queries)
            logger.info(f"After evolution: {len(base_queries)} queries")

        # Step 3: Final deduplication and validation
        final_queries = self._final_filter(base_queries)
        logger.info(f"After final filtering: {len(final_queries)} queries")

        # Step 4: Trim to target count
        result = final_queries[:target_count]
        logger.info(f"Final result: {len(result)} queries (target: {target_count})")

        return result

    def _is_duplicate(self, query: GeneratedQuery, existing: List[GeneratedQuery]) -> bool:
        """Check if a query is duplicate of existing queries (simple text comparison)."""
        query_text = query.query.strip().lower()
        for eq in existing:
            if query_text == eq.query.strip().lower():
                return True
        return False

    # =========================================================================
    # Parallel Batch Generation
    # =========================================================================

    async def _parallel_generate(self) -> List[GeneratedQuery]:
        """Generate queries in parallel batches for better diversity."""
        num_batches = self.query_config.num_parallel_batches
        queries_per_call = self.query_config.queries_per_call

        # Calculate target per batch, respecting queries_per_call limit
        # Generate extra to account for deduplication
        ideal_per_batch = (self.query_config.num_queries * 2) // num_batches + 1
        target_per_batch = min(ideal_per_batch, queries_per_call)

        logger.info(
            f"Launching {num_batches} parallel generation batches, "
            f"{target_per_batch} queries each (queries_per_call={queries_per_call})"
        )

        # Create tasks for parallel execution
        tasks = [self._generate_batch(batch_id=i, num_queries=target_per_batch) for i in range(num_batches)]

        # Execute in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect results, handling any errors
        all_queries: List[GeneratedQuery] = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning(f"Batch {i} failed: {result}")
            else:
                all_queries.extend(result)

        return all_queries

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    async def _generate_batch(self, batch_id: int, num_queries: int) -> List[GeneratedQuery]:
        """Generate a single batch of queries.

        Args:
            batch_id: Batch identifier for diversity
            num_queries: Number of queries to generate

        Returns:
            List of GeneratedQuery objects
        """
        # Format seed queries
        seed_queries_text = "None provided"
        if self.query_config.seed_queries:
            seed_queries_text = "\n".join(f"- {q}" for q in self.query_config.seed_queries)

        # Format categories
        categories_text = "No specific categories, generate diverse queries"
        if self.query_config.categories:
            categories_text = "\n".join(
                f"- {cat.get('name', 'unknown')}: weight {cat.get('weight', 1.0)}"
                for cat in self.query_config.categories
            )

        # Get existing queries context
        existing_context = self.deduplicator.get_existing_summary(max_items=15)

        # Build prompt
        messages = QUERY_GENERATION_TEMPLATE.format(
            task_description=self.task_config.description,
            scenario=self.task_config.scenario or "General usage",
            seed_queries=seed_queries_text,
            categories=categories_text,
            existing_queries=existing_context,
            num_queries=num_queries,
            batch_id=batch_id,
        )

        # Call model with structured output
        response = await self.model.achat(
            messages=list(messages),
            structured_model=QueryGenerationOutput,
        )

        if not response.parsed or "queries" not in response.parsed:
            raise ValueError(f"Failed to parse query generation response for batch {batch_id}")

        # Parse and deduplicate queries
        queries: List[GeneratedQuery] = []
        for q in response.parsed["queries"]:
            if isinstance(q, dict):
                query_obj = GeneratedQuery(**q)
            else:
                query_obj = q

            # Validate and deduplicate
            is_valid, reason = QueryValidator.validate(query_obj)
            if not is_valid:
                logger.debug(f"Batch {batch_id}: Skipping invalid query: {reason}")
                continue

            if self.deduplicator.add(query_obj.query):
                queries.append(query_obj)
            else:
                logger.debug(f"Batch {batch_id}: Skipping duplicate query")

        logger.info(f"Batch {batch_id}: Generated {len(queries)} valid unique queries")
        return queries

    # =========================================================================
    # Evol-Instruct Complexity Evolution
    # =========================================================================

    async def _evolve_queries(self, base_queries: List[GeneratedQuery]) -> List[GeneratedQuery]:
        """Apply Evol-Instruct style complexity evolution to queries.

        Args:
            base_queries: Base queries to evolve

        Returns:
            List of evolved queries
        """
        if not base_queries:
            return []

        # Select queries for evolution (prefer easier ones and seed queries)
        candidates = self._select_evolution_candidates(base_queries)
        logger.info(f"Selected {len(candidates)} queries for evolution")

        evolved_queries: List[GeneratedQuery] = []

        for round_idx in range(self.query_config.evolution_rounds):
            logger.info(f"Evolution round {round_idx + 1}/{self.query_config.evolution_rounds}")

            # Create evolution tasks for each strategy
            tasks = []
            for query in candidates:
                for strategy in self.query_config.complexity_levels:
                    tasks.append(self._evolve_single(query, strategy))

            # Execute evolutions in parallel (with some concurrency limit)
            semaphore = asyncio.Semaphore(5)

            async def limited_evolve(task):
                async with semaphore:
                    return await task

            results = await asyncio.gather(*[limited_evolve(t) for t in tasks], return_exceptions=True)

            # Collect results
            for result in results:
                if isinstance(result, Exception):
                    logger.debug(f"Evolution failed: {result}")
                elif result:
                    evolved_queries.extend(result)

            # Update candidates for next round
            if evolved_queries:
                candidates = self._select_evolution_candidates(evolved_queries[-10:])

        return evolved_queries

    def _select_evolution_candidates(
        self, queries: List[GeneratedQuery], max_candidates: int = 5
    ) -> List[GeneratedQuery]:
        """Select best candidates for evolution."""
        # Prefer easier queries and shorter ones for evolution
        scored = []
        for q in queries:
            score = 0
            if q.difficulty == "easy":
                score += 2
            elif q.difficulty == "medium":
                score += 1
            # Prefer medium-length queries
            length = len(q.query)
            if 20 <= length <= 100:
                score += 1
            scored.append((score, q))

        scored.sort(key=lambda x: -x[0])
        return [q for _, q in scored[:max_candidates]]

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=0.5, min=0.5, max=5))
    async def _evolve_single(self, query: GeneratedQuery, strategy: str) -> List[GeneratedQuery]:
        """Evolve a single query using the specified strategy.

        Args:
            query: Query to evolve
            strategy: Evolution strategy (constraints, reasoning, edge_cases, combination)

        Returns:
            List of evolved queries
        """
        messages = EVOLUTION_TEMPLATE.format(
            original_query=query.query,
            strategy=strategy,
            num_variations=2,
        )

        try:
            response = await self.model.achat(
                messages=list(messages),
                structured_model=EvolutionOutput,
            )

            if not response.parsed or "evolved_queries" not in response.parsed:
                return []

            evolved: List[GeneratedQuery] = []
            for eq in response.parsed["evolved_queries"]:
                if isinstance(eq, dict):
                    evolved_query = GeneratedQuery(
                        query=eq.get("query", ""),
                        category=query.category,  # Inherit category
                        difficulty=eq.get("difficulty", "hard"),
                    )
                else:
                    evolved_query = GeneratedQuery(
                        query=eq.query,
                        category=query.category,
                        difficulty=eq.difficulty,
                    )

                # Validate and deduplicate
                is_valid, _ = QueryValidator.validate(evolved_query)
                if is_valid and self.deduplicator.add(evolved_query.query):
                    evolved.append(evolved_query)

            return evolved

        except Exception as e:
            logger.debug(f"Evolution failed for strategy {strategy}: {e}")
            return []

    # =========================================================================
    # Final Filtering and Balancing
    # =========================================================================

    def _final_filter(self, queries: List[GeneratedQuery]) -> List[GeneratedQuery]:
        """Apply final filtering and category balancing.

        Args:
            queries: All generated queries

        Returns:
            Filtered and balanced queries
        """
        # Re-validate all queries
        valid_queries = []
        for q in queries:
            is_valid, _ = QueryValidator.validate(q)
            if is_valid:
                valid_queries.append(q)

        # Apply category balancing if categories specified
        if self.query_config.categories:
            return self._balance_categories(valid_queries)

        return valid_queries

    def _balance_categories(self, queries: List[GeneratedQuery]) -> List[GeneratedQuery]:
        """Balance queries according to category weights."""
        if not self.query_config.categories:
            return queries

        # Calculate target counts per category
        total_weight = sum(c.get("weight", 1.0) for c in self.query_config.categories)
        target_counts: Dict[str, int] = {}
        for cat in self.query_config.categories:
            name = cat.get("name", "general")
            weight = cat.get("weight", 1.0)
            target_counts[name] = max(1, int(self.query_config.num_queries * weight / total_weight))

        # Group queries by category
        by_category: Dict[str, List[GeneratedQuery]] = defaultdict(list)
        uncategorized: List[GeneratedQuery] = []

        for q in queries:
            if q.category and q.category in target_counts:
                by_category[q.category].append(q)
            else:
                uncategorized.append(q)

        # Build balanced result
        result: List[GeneratedQuery] = []

        for cat, target in target_counts.items():
            available = by_category.get(cat, [])
            result.extend(available[:target])

        # Fill remaining with uncategorized
        remaining = self.query_config.num_queries - len(result)
        result.extend(uncategorized[:remaining])

        return result
