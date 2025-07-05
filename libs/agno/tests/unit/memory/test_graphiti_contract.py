"""
Contract tests for GraphitiMemoryDb implementation.

This runs the standard MemoryDb contract tests against GraphitiMemoryDb
to ensure it correctly implements the interface.
"""

from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch

import pytest

from agno.memory.v2.db.schema import MemoryRow
from tests.unit.memory.test_memory_contract import MemoryDbContractTests


class TestGraphitiMemoryDbContract(MemoryDbContractTests):
    """Run contract tests against GraphitiMemoryDb implementation."""

    @pytest.fixture
    def memory_db(self):
        """Provide a mocked GraphitiMemoryDb instance for contract testing."""
        with patch("agno.memory.v2.db.graphiti.Graphiti") as mock_graphiti_class:
            # Import here to avoid import errors when graphiti-core is not installed
            from agno.memory.v2.db.graphiti import GraphitiMemoryDb

            # Create mock Graphiti instance
            mock_graphiti = Mock()
            mock_graphiti_class.return_value = mock_graphiti

            # Setup async method mocks
            mock_graphiti.build_indices_for_schema = AsyncMock(return_value=None)
            mock_graphiti.retrieve_episodes = AsyncMock(return_value=[])
            mock_graphiti.search = AsyncMock(return_value=[])
            mock_graphiti.add_episode = AsyncMock(return_value=None)
            mock_graphiti.remove_episode = AsyncMock(return_value=None)

            # Create GraphitiMemoryDb instance
            db = GraphitiMemoryDb(
                uri="bolt://localhost:7687", user="neo4j", password="password", table_name="test_contract"
            )

            # Track state for mocking
            self._episodes = {}

            # Override async methods to simulate behavior
            def mock_add_episode(**kwargs):
                episode_name = kwargs.get("name")
                self._episodes[episode_name] = {
                    "name": episode_name,
                    "episode_body": kwargs.get("episode_body"),
                    "source_description": kwargs.get("source_description"),
                    "reference_time": kwargs.get("reference_time"),
                    "group_id": kwargs.get("group_id"),
                }
                return None

            def mock_remove_episode(episode_name):
                self._episodes.pop(episode_name, None)
                return None

            def mock_retrieve_episodes(**kwargs):
                episode_names = kwargs.get("episode_names", [])
                if episode_names:
                    return [self._episodes.get(name) for name in episode_names if name in self._episodes]
                return list(self._episodes.values())

            def mock_search(**kwargs):
                # Simple search implementation
                results = []
                group_ids = kwargs.get("group_ids", [])
                num_results = kwargs.get("num_results", 1000)

                for episode in self._episodes.values():
                    # Filter by group_id if specified
                    if group_ids and episode.get("group_id") not in group_ids:
                        continue

                    # Filter by source_description (table isolation)
                    if not episode.get("source_description", "").startswith("agno_memory_test_contract"):
                        continue

                    results.append(episode)

                # Limit results
                return results[:num_results]

            # Set up the mocks
            mock_graphiti.add_episode = AsyncMock(side_effect=mock_add_episode)
            mock_graphiti.remove_episode = AsyncMock(side_effect=mock_remove_episode)
            mock_graphiti.retrieve_episodes = AsyncMock(side_effect=mock_retrieve_episodes)
            mock_graphiti.search = AsyncMock(side_effect=mock_search)

            return db

    def test_similarity_search_contract(self, memory_db):
        """Test that GraphitiMemoryDb supports similarity search."""
        memory_db.create()

        # Insert some memories
        memories = [
            MemoryRow(
                id=f"mem_{i}",
                memory={"content": f"Memory about {topic}"},
                user_id="test_user",
                last_updated=datetime.now(),
            )
            for i, topic in enumerate(["cats", "dogs", "birds"])
        ]

        for memory in memories:
            memory_db.upsert_memory(memory)

        # Similarity search should return results
        results = memory_db.similarity_search("pets")
        assert isinstance(results, list)

        # With user filter
        results = memory_db.similarity_search("animals", user_id="test_user")
        assert isinstance(results, list)

        # With limit
        results = memory_db.similarity_search("creatures", limit=2)
        assert isinstance(results, list)
        assert len(results) <= 2

    def test_bulk_operations_contract(self, memory_db):
        """Test that GraphitiMemoryDb supports bulk operations."""
        memory_db.create()

        # Create test memories
        memories = [
            MemoryRow(
                id=f"bulk_{i}", memory={"content": f"Bulk memory {i}"}, user_id="bulk_user", last_updated=datetime.now()
            )
            for i in range(5)
        ]

        # Bulk upsert
        result = memory_db.bulk_upsert_memories(memories)
        assert isinstance(result, list)
        assert len(result) == len(memories)

        # Verify all were inserted
        all_memories = memory_db.read_memories(user_id="bulk_user")
        assert len(all_memories) == len(memories)

        # Bulk delete
        memory_ids = [m.id for m in memories]
        deleted_count = memory_db.bulk_delete_memories(memory_ids)
        assert deleted_count == len(memories)

        # Verify all were deleted
        remaining = memory_db.read_memories(user_id="bulk_user")
        assert len(remaining) == 0
