"""
GraphitiMemoryDb: Graphiti-based implementation of the MemoryDb interface.

This module provides a graph database backend for Agno's memory system using Graphiti,
enabling graph-based memory storage with vector similarity search and relationship modeling.
"""

import asyncio
import json
import logging
import re
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from agno.memory.v2.db.base import MemoryDb
from agno.memory.v2.db.schema import MemoryRow

try:
    from graphiti_core import Graphiti
    from graphiti_core.embedder.client import EmbedderClient
    from graphiti_core.llm_client.client import LLMClient
except ImportError:
    raise ImportError(
        "The 'graphiti-core' package is required to use GraphitiMemoryDb. Install it with: pip install agno[graphiti]"
    )


class GraphitiMemoryDb(MemoryDb):
    """
    Graphiti-based memory database implementation.

    This class implements the MemoryDb interface using Graphiti's graph database
    capabilities, providing vector similarity search and relationship modeling
    for memory storage.
    """

    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        table_name: str = "agno_memories",
        llm_client: Optional[LLMClient] = None,
        embedder: Optional[EmbedderClient] = None,
        max_results: int = 1000,
        max_json_size_mb: int = 1,
        **kwargs,
    ):
        """
        Initialize GraphitiMemoryDb.

        Args:
            uri: Neo4j connection URI (bolt:// or neo4j://)
            user: Database username
            password: Database password
            table_name: Logical table name for episode grouping
            llm_client: Optional custom LLM client
            embedder: Optional custom embedder client
            max_results: Maximum number of results to return in search operations (default: 1000)
            max_json_size_mb: Maximum size of JSON payloads in MB for security (default: 1MB)
            **kwargs: Additional arguments passed to Graphiti constructor

        Note:
            The max_results and max_json_size_mb parameters provide configurable limits
            for performance and security. Adjust based on your use case requirements.
        """
        self.uri = uri
        self.user = user
        self.password = password
        self.table_name = table_name

        # Validate configuration parameters
        if max_results <= 0:
            raise ValueError("max_results must be positive")
        if max_json_size_mb <= 0:
            raise ValueError("max_json_size_mb must be positive")

        self.max_results = max_results
        self.max_json_size_mb = max_json_size_mb

        # Initialize Graphiti client
        self.graphiti = Graphiti(
            uri=uri, user=user, password=password, llm_client=llm_client, embedder=embedder, **kwargs
        )

        # Group prefix for organizing memories by table
        self.group_prefix = f"agno_{table_name}"

        # Configure logging
        self.logger = logging.getLogger(__name__)

        # Retry configuration
        self.max_retries = 3
        self.backoff_factor = 2

    def _run_async(self, coro):
        """
        Safely run async coroutine in sync context.

        Handles both cases where we're already in an event loop
        or need to create a new one without causing deadlocks.
        """
        try:
            # Check if we're already in an event loop
            asyncio.get_running_loop()
            # If we're in a loop, use a new thread without an event loop
            import concurrent.futures

            def run_in_thread():
                # Create a new event loop in this thread
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    return new_loop.run_until_complete(coro)
                finally:
                    new_loop.close()
                    asyncio.set_event_loop(None)

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_in_thread)
                return future.result()
        except RuntimeError:
            # No running loop, safe to use asyncio.run()
            return asyncio.run(coro)

    def _run_async_with_retry(self, coro, operation_name: str = "operation"):
        """
        Run async coroutine with retry logic for transient failures.

        Args:
            coro: Async coroutine to execute
            operation_name: Name of operation for logging

        Returns:
            Result of the coroutine execution
        """
        for attempt in range(self.max_retries):
            try:
                return self._run_async(coro)
            except Exception as e:
                # Check if this is a retriable error
                if self._is_retriable_error(e) and attempt < self.max_retries - 1:
                    wait_time = self.backoff_factor**attempt
                    self.logger.warning(
                        f"{operation_name} failed (attempt {attempt + 1}/{self.max_retries}), "
                        f"retrying in {wait_time} seconds: {e}"
                    )
                    time.sleep(wait_time)
                    continue
                else:
                    # Not retriable or max retries reached
                    if attempt == self.max_retries - 1:
                        self.logger.error(f"{operation_name} failed after {self.max_retries} attempts: {e}")
                    raise

        # Should never reach here
        raise RuntimeError(f"{operation_name} failed after {self.max_retries} attempts")

    def _is_retriable_error(self, error: Exception) -> bool:
        """
        Determine if an error is retriable (transient).

        Args:
            error: Exception to check

        Returns:
            bool: True if error should be retried
        """
        error_str = str(error).lower()

        # Common transient error patterns
        transient_patterns = [
            "connection",
            "timeout",
            "network",
            "temporary",
            "unavailable",
            "server error",
            "503",
            "502",
            "504",
            "reset",
            "refused",
        ]

        return any(pattern in error_str for pattern in transient_patterns)

    def create(self) -> None:
        """
        Initialize the memory database.

        For Graphiti, this ensures the connection is established and
        any necessary indices are created.
        """
        try:
            # Verify connection by attempting to initialize client
            self._run_async_with_retry(self.graphiti.build_indices_for_schema(), "database initialization")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Graphiti database: {e}") from e

    def memory_exists(self, memory: MemoryRow) -> bool:
        """
        Check if a memory with the given ID already exists.

        Args:
            memory: MemoryRow object to check

        Returns:
            bool: True if memory exists, False otherwise
        """
        if memory.id is None:
            return False

        try:
            episode_name = f"memory_{memory.id}"
            episodes = self._run_async(self.graphiti.retrieve_episodes(episode_names=[episode_name]))
            return len(episodes) > 0
        except Exception as e:
            self.logger.warning(f"Failed to check memory existence for {memory.id}: {e}")
            return False

    def read_memories(
        self,
        user_id: Optional[str] = None,
        limit: Optional[int] = None,
        sort: Optional[str] = None,
    ) -> List[MemoryRow]:
        """
        Retrieve memories with optional filtering and sorting.

        Args:
            user_id: Filter by user (optional)
            limit: Maximum number of records to return (optional)
            sort: Sort order - "asc" or "desc" (optional, defaults to desc)

        Returns:
            List[MemoryRow]: List of memory records
        """
        try:
            # Build optimized search parameters
            search_params = self._build_episode_search_query(user_id=user_id, limit=limit)
            search_params = self._optimize_search_parameters(search_params)

            # Retrieve episodes using search
            episodes = self._run_async(self.graphiti.search(**search_params))

            # Convert episodes to MemoryRow objects with validation
            memories = []
            for episode in episodes:
                try:
                    # Validate episode belongs to this table
                    if self._validate_episode_belongs_to_table(episode):
                        memory = self._episode_to_memory(episode)
                        memories.append(memory)
                except Exception as e:
                    # Skip episodes that can't be converted but log the issue
                    self.logger.warning(f"Failed to convert episode {episode.get('name', 'unknown')}: {e}")
                    continue

            # Sort by last_updated if requested
            if sort == "asc":
                memories.sort(key=lambda m: m.last_updated or datetime.min)
            else:
                # Default to descending order
                memories.sort(key=lambda m: m.last_updated or datetime.min, reverse=True)

            # Apply limit if not already handled by Graphiti
            if limit is not None and len(memories) > limit:
                memories = memories[:limit]

            return memories

        except Exception as e:
            raise RuntimeError(f"Failed to read memories: {e}") from e

    def upsert_memory(self, memory: MemoryRow) -> Optional[MemoryRow]:
        """
        Insert or update a memory record.

        Args:
            memory: MemoryRow object to upsert

        Returns:
            Optional[MemoryRow]: The upserted memory (or None based on interface)
        """
        try:
            # Convert to episode format
            episode_data = self._memory_to_episode(memory)

            # Check if episode already exists
            exists = self.memory_exists(memory)

            if exists:
                # For updates, remove existing episode first
                self._run_async_with_retry(
                    self.graphiti.remove_episode(episode_data["name"]), f"remove episode {episode_data['name']}"
                )

            # Add the episode
            self._run_async_with_retry(self.graphiti.add_episode(**episode_data), f"add episode {episode_data['name']}")

            # Return the memory with updated id and timestamp
            return memory

        except Exception as e:
            raise RuntimeError(f"Failed to upsert memory: {e}") from e

    def delete_memory(self, memory_id: str) -> None:
        """
        Delete a memory by ID.

        Args:
            memory_id: String ID of memory to delete
        """
        try:
            episode_name = f"memory_{memory_id}"
            self._run_async_with_retry(self.graphiti.remove_episode(episode_name), f"delete memory {memory_id}")
        except Exception as e:
            raise RuntimeError(f"Failed to delete memory {memory_id}: {e}") from e

    def drop_table(self) -> None:
        """
        Drop/delete the entire memory table/collection.

        For Graphiti, this removes all episodes associated with this table.
        """
        try:
            # Get all memories for this table (optimized to just get IDs)
            search_params = self._build_episode_search_query()
            search_params = self._optimize_search_parameters(search_params)
            episodes = self._run_async(self.graphiti.search(**search_params))

            # Extract episode names and delete in batch
            episode_names = []
            for episode in episodes:
                if self._validate_episode_belongs_to_table(episode):
                    episode_names.append(episode.get("name"))

            # Delete all episodes (Graphiti doesn't have bulk delete, so we batch them)
            if episode_names:

                async def bulk_delete():
                    tasks = [self.graphiti.remove_episode(name) for name in episode_names]
                    await asyncio.gather(*tasks, return_exceptions=True)

                self._run_async(bulk_delete())

        except Exception as e:
            raise RuntimeError(f"Failed to drop table {self.table_name}: {e}") from e

    def table_exists(self) -> bool:
        """
        Check if the memory table/collection exists.

        Returns:
            bool: True if table exists (has any episodes), False otherwise
        """
        try:
            # Check if any memories exist for this table
            memories = self.read_memories(limit=1)
            return len(memories) > 0
        except Exception:
            return False

    def clear(self) -> bool:
        """
        Remove all records from the memory table.

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.drop_table()
            return True
        except Exception:
            return False

    def similarity_search(
        self, query: str, user_id: Optional[str] = None, limit: Optional[int] = None
    ) -> List[MemoryRow]:
        """
        Perform vector similarity search on memories.

        Args:
            query: Search query text
            user_id: Filter by user (optional)
            limit: Maximum number of results (optional)

        Returns:
            List[MemoryRow]: Ranked list of similar memories
        """
        try:
            # Build search parameters for similarity search
            search_params = self._build_episode_search_query(user_id=user_id, content_query=query, limit=limit)
            search_params = self._optimize_search_parameters(search_params)

            # Execute similarity search
            episodes = self._run_async(self.graphiti.search(**search_params))

            # Convert and validate results
            memories = []
            for episode in episodes:
                try:
                    if self._validate_episode_belongs_to_table(episode):
                        memory = self._episode_to_memory(episode)
                        memories.append(memory)
                except Exception as e:
                    self.logger.warning(
                        f"Failed to convert episode {episode.get('name', 'unknown')} in similarity search: {e}"
                    )
                    continue

            return memories

        except Exception as e:
            raise RuntimeError(f"Failed to perform similarity search: {e}") from e

    def bulk_upsert_memories(self, memories: List[MemoryRow]) -> List[MemoryRow]:
        """
        Efficiently upsert multiple memories in batch.

        Args:
            memories: List of MemoryRow objects to upsert

        Returns:
            List[MemoryRow]: List of successfully upserted memories
        """
        try:
            # Build bulk operation parameters
            episode_params = self._build_bulk_operation_params(memories, "upsert")

            # Batch check for existing episodes to avoid N+1 pattern
            episode_names = [ep["name"] for ep in episode_params]
            existing_episodes = self._run_async(self.graphiti.retrieve_episodes(episode_names=episode_names))
            existing_names = {ep.get("name") for ep in existing_episodes}

            # Process each memory individually to track success/failure
            async def bulk_upsert():
                successful_indices = []

                for i, episode_data in enumerate(episode_params):
                    try:
                        # Remove if exists
                        if episode_data["name"] in existing_names:
                            await self.graphiti.remove_episode(episode_data["name"])

                        # Add the episode
                        await self.graphiti.add_episode(**episode_data)
                        successful_indices.append(i)  # Track successful memory index

                    except Exception as e:
                        self.logger.warning(f"Failed to upsert memory {i}: {e}")
                        continue

                return successful_indices

            successful_indices = self._run_async(bulk_upsert())
            successful_memories = [memories[i] for i in successful_indices]

            return successful_memories

        except Exception as e:
            raise RuntimeError(f"Failed to bulk upsert memories: {e}") from e

    def bulk_delete_memories(self, memory_ids: List[str]) -> int:
        """
        Efficiently delete multiple memories in batch.

        Args:
            memory_ids: List of memory IDs to delete

        Returns:
            int: Number of successfully deleted memories
        """
        try:
            # Batch delete operations
            episode_names = [f"memory_{memory_id}" for memory_id in memory_ids]

            async def bulk_delete():
                tasks = [self.graphiti.remove_episode(name) for name in episode_names]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                # Count successful deletions (not exceptions)
                return sum(1 for result in results if not isinstance(result, Exception))

            deleted_count = self._run_async(bulk_delete())
            return deleted_count

        except Exception as e:
            raise RuntimeError(f"Failed to bulk delete memories: {e}") from e

    # Private helper methods

    def _memory_to_episode(self, memory: MemoryRow) -> Dict[str, Any]:
        """
        Convert a MemoryRow to Graphiti episode format.

        Args:
            memory: MemoryRow to convert

        Returns:
            Dict[str, Any]: Episode data for Graphiti
        """
        # Generate UUID if not present (without mutating input during conversion)
        memory_id = memory.id if memory.id is not None else str(uuid.uuid4())

        # Set timestamp if not present (without mutating input during conversion)
        last_updated = memory.last_updated if memory.last_updated is not None else datetime.now()

        # Build episode data first
        episode_data = {
            "name": f"memory_{memory_id}",
            "episode_body": json.dumps(memory.memory),
            "source_description": f"agno_memory_{self.table_name}",
            "reference_time": last_updated,
            "group_id": self._generate_group_id(memory.user_id) if memory.user_id else None,
        }

        # Only update the memory object after successful conversion to maintain consistency
        if memory.id is None:
            memory.id = memory_id
        if memory.last_updated is None:
            memory.last_updated = last_updated

        return episode_data

    def _episode_to_memory(self, episode: Dict[str, Any]) -> MemoryRow:
        """
        Convert a Graphiti episode back to MemoryRow format.

        Args:
            episode: Episode data from Graphiti

        Returns:
            MemoryRow: Converted memory record
        """
        # Extract memory ID from episode name (format: "memory_{id}")
        episode_name = episode.get("name", "")
        memory_id = episode_name.replace("memory_", "") if episode_name.startswith("memory_") else None

        # Deserialize memory data from episode body with security validation
        memory_data = {}
        if episode.get("episode_body"):
            try:
                episode_body = episode["episode_body"]

                # Security: Limit JSON size to prevent memory exhaustion
                max_size_bytes = self.max_json_size_mb * 1024 * 1024
                if len(episode_body) > max_size_bytes:
                    raise ValueError(
                        f"Episode body too large: {len(episode_body)} bytes (max {self.max_json_size_mb}MB)"
                    )

                memory_data = json.loads(episode_body)

                # Validate that result is a dictionary
                if not isinstance(memory_data, dict):
                    raise ValueError("Episode body must deserialize to a dictionary")

            except json.JSONDecodeError as e:
                self.logger.warning(f"Invalid JSON in episode body, treating as plain content: {e}")
                memory_data = {"content": episode["episode_body"]}
            except ValueError as e:
                self.logger.error(f"Security validation failed for episode body: {e}")
                memory_data = {"content": "INVALID_DATA"}

        # Extract user_id from group_id
        user_id = None
        if episode.get("group_id"):
            user_id = self._extract_user_id(episode["group_id"])

        # Convert reference_time to datetime
        last_updated = episode.get("reference_time")
        if isinstance(last_updated, str):
            try:
                last_updated = datetime.fromisoformat(last_updated.replace("Z", "+00:00"))
            except ValueError:
                last_updated = None

        return MemoryRow(id=memory_id, memory=memory_data, user_id=user_id, last_updated=last_updated)

    def _translate_filters(self, user_id: Optional[str] = None, limit: Optional[int] = None) -> Dict[str, Any]:
        """
        Translate Agno filter parameters to Graphiti search parameters.

        Args:
            user_id: User filter
            limit: Result limit

        Returns:
            Dict[str, Any]: Graphiti search parameters
        """
        filters = {}

        if user_id is not None:
            filters["group_ids"] = [self._generate_group_id(user_id)]

        if limit is not None:
            filters["num_results"] = limit

        return filters

    def _generate_group_id(self, user_id: str) -> str:
        """
        Generate group_id for a user within this table context.

        Args:
            user_id: User identifier

        Returns:
            str: Formatted group_id

        Raises:
            ValueError: If user_id contains invalid characters
        """
        # Validate user_id to prevent injection and formatting issues
        if not user_id:
            raise ValueError("user_id cannot be empty")

        # Allow alphanumeric, underscore, hyphen, and period characters
        if not re.match(r"^[a-zA-Z0-9_.-]{1,64}$", user_id):
            raise ValueError(
                f"Invalid user_id format: {user_id}. Must be 1-64 characters containing only letters, numbers, underscore, hyphen, or period"
            )

        return f"{self.group_prefix}_user_{user_id}"

    def _extract_user_id(self, group_id: str) -> Optional[str]:
        """
        Extract user_id from a group_id.

        Args:
            group_id: Graphiti group identifier

        Returns:
            Optional[str]: Extracted user_id or None if not parseable
        """
        prefix = f"{self.group_prefix}_user_"
        if group_id.startswith(prefix):
            return group_id[len(prefix) :]
        return None

    # Cypher Query Builder Utilities

    def _build_episode_search_query(
        self,
        user_id: Optional[str] = None,
        memory_ids: Optional[List[str]] = None,
        content_query: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Build search parameters for episode retrieval.

        Args:
            user_id: Filter by user
            memory_ids: Specific memory IDs to search for
            content_query: Text query for semantic search
            limit: Maximum results

        Returns:
            Dict containing search parameters for Graphiti
        """
        search_params = {}

        # Handle user filtering via group_ids
        if user_id is not None:
            search_params["group_ids"] = [self._generate_group_id(user_id)]

        # Handle specific episode name filtering
        if memory_ids is not None:
            episode_names = [f"memory_{mid}" for mid in memory_ids]
            search_params["episode_names"] = episode_names

        # Handle content-based search
        if content_query is not None:
            search_params["query"] = content_query
        else:
            # Default empty query to get all episodes
            search_params["query"] = ""

        # Handle result limiting
        if limit is not None:
            search_params["num_results"] = limit

        return search_params

    def _build_episode_filter_criteria(
        self, table_name_filter: bool = True, additional_filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Build filter criteria for episodes to ensure table isolation.

        Args:
            table_name_filter: Whether to filter by table name
            additional_filters: Additional filter criteria

        Returns:
            Dict containing filter criteria
        """
        criteria = {}

        if table_name_filter:
            # Filter by source_description to ensure table isolation
            criteria["source_description_prefix"] = f"agno_memory_{self.table_name}"

        if additional_filters:
            criteria.update(additional_filters)

        return criteria

    def _validate_episode_belongs_to_table(self, episode: Dict[str, Any]) -> bool:
        """
        Validate that an episode belongs to this table.

        Args:
            episode: Episode data from Graphiti

        Returns:
            bool: True if episode belongs to this table
        """
        source_desc = episode.get("source_description", "")
        expected_prefix = f"agno_memory_{self.table_name}"
        return source_desc.startswith(expected_prefix)

    def _build_bulk_operation_params(
        self, memories: List[MemoryRow], operation_type: str = "upsert"
    ) -> List[Dict[str, Any]]:
        """
        Build parameters for bulk operations on multiple memories.

        Args:
            memories: List of MemoryRow objects
            operation_type: Type of operation ("upsert", "delete")

        Returns:
            List of parameter dictionaries for batch processing
        """
        if operation_type == "upsert":
            return [self._memory_to_episode(memory) for memory in memories]
        elif operation_type == "delete":
            return [{"episode_name": f"memory_{memory.id}"} for memory in memories if memory.id]
        else:
            raise ValueError(f"Unsupported operation type: {operation_type}")

    def _optimize_search_parameters(self, search_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Optimize search parameters for better performance.

        Args:
            search_params: Raw search parameters

        Returns:
            Optimized search parameters
        """
        optimized = search_params.copy()

        # Ensure reasonable limits to prevent memory issues
        if "num_results" not in optimized or optimized.get("num_results", 0) > self.max_results:
            optimized["num_results"] = self.max_results

        # Add table-specific filtering if not present
        if "group_ids" not in optimized:
            # Add a hint to help with query planning
            optimized["_table_context"] = self.table_name

        return optimized
