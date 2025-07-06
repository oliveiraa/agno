"""
Unit tests for GraphitiMemoryDb implementation.

Tests the Graphiti-based memory database provider with mocked dependencies,
focusing on CRUD operations, filter translation, and interface compliance.
"""

import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agno.memory.v2.db.graphiti import GraphitiMemoryDb
from agno.memory.v2.db.schema import MemoryRow


@pytest.fixture
def mock_graphiti_client():
    """Mock Graphiti client with async methods."""
    with patch("agno.memory.v2.db.graphiti.Graphiti") as mock_graphiti:
        client = MagicMock()

        # Mock async methods
        client.build_indices_for_schema = AsyncMock()
        client.add_episode = AsyncMock()
        client.search = AsyncMock(return_value=[])
        client.retrieve_episodes = AsyncMock(return_value=[])
        client.remove_episode = AsyncMock()

        # Return the mock client when Graphiti is instantiated
        mock_graphiti.return_value = client
        yield client


@pytest.fixture
def memory_db(mock_graphiti_client):
    """Create GraphitiMemoryDb instance with mocked client."""
    return GraphitiMemoryDb(uri="neo4j://localhost:7687", user="neo4j", password="password", table_name="test_memories")


@pytest.fixture
def sample_memory_row():
    """Create a sample MemoryRow for testing."""
    return MemoryRow(
        id="test-memory-123",
        memory={"content": "User likes pizza", "topics": ["food", "preferences"]},
        user_id="test_user",
        last_updated=datetime(2024, 1, 1, 12, 0, 0),
    )


@pytest.fixture
def sample_episode():
    """Create a sample Graphiti episode for testing."""
    return {
        "name": "memory_test-memory-123",
        "episode_body": json.dumps({"content": "User likes pizza", "topics": ["food", "preferences"]}),
        "source_description": "agno_memory_test_memories",
        "reference_time": datetime(2024, 1, 1, 12, 0, 0),
        "group_id": "agno_test_memories_user_test_user",
    }


class TestGraphitiMemoryDbInitialization:
    """Test initialization and configuration."""

    def test_initialization_with_required_parameters(self, mock_graphiti_client):
        """Test successful initialization with required parameters."""
        db = GraphitiMemoryDb(uri="neo4j://localhost:7687", user="neo4j", password="password")

        assert db.uri == "neo4j://localhost:7687"
        assert db.user == "neo4j"
        assert db.password == "password"
        assert db.table_name == "agno_memories"  # default
        assert db.group_prefix == "agno_agno_memories"

    def test_initialization_with_custom_table_name(self, mock_graphiti_client):
        """Test initialization with custom table name."""
        db = GraphitiMemoryDb(
            uri="neo4j://localhost:7687", user="neo4j", password="password", table_name="custom_table"
        )

        assert db.table_name == "custom_table"
        assert db.group_prefix == "agno_custom_table"

    def test_graphiti_client_initialization(self, mock_graphiti_client):
        """Test that Graphiti client is properly initialized."""
        from agno.memory.v2.db.graphiti import Graphiti

        GraphitiMemoryDb(uri="neo4j://localhost:7687", user="neo4j", password="password")

        # Verify Graphiti was called with correct parameters
        Graphiti.assert_called_once_with(
            uri="neo4j://localhost:7687", user="neo4j", password="password", llm_client=None, embedder=None
        )


class TestCreateMethod:
    """Test database initialization."""

    def test_create_success(self, memory_db, mock_graphiti_client):
        """Test successful database creation."""
        memory_db.create()

        # Verify build_indices_for_schema was called
        mock_graphiti_client.build_indices_for_schema.assert_called_once()

    def test_create_failure(self, memory_db, mock_graphiti_client):
        """Test database creation failure handling."""
        mock_graphiti_client.build_indices_for_schema.side_effect = Exception("Connection failed")

        with pytest.raises(RuntimeError, match="Failed to initialize Graphiti database"):
            memory_db.create()


class TestMemoryConversionHelpers:
    """Test helper methods for memory/episode conversion."""

    def test_memory_to_episode_conversion(self, memory_db, sample_memory_row):
        """Test converting MemoryRow to Graphiti episode format."""
        episode = memory_db._memory_to_episode(sample_memory_row)

        assert episode["name"] == "memory_test-memory-123"
        assert episode["episode_body"] == json.dumps(sample_memory_row.memory)
        assert episode["source_description"] == "agno_memory_test_memories"
        assert episode["reference_time"] == sample_memory_row.last_updated
        assert episode["group_id"] == "agno_test_memories_user_test_user"

    def test_memory_to_episode_with_auto_id_generation(self, memory_db):
        """Test episode conversion with automatic ID generation."""
        memory_without_id = MemoryRow(id=None, memory={"content": "Test content"}, user_id="test_user")

        episode = memory_db._memory_to_episode(memory_without_id)

        # ID should be generated
        assert memory_without_id.id is not None
        assert episode["name"] == f"memory_{memory_without_id.id}"

    def test_memory_to_episode_with_auto_timestamp(self, memory_db):
        """Test episode conversion with automatic timestamp generation."""
        memory_without_timestamp = MemoryRow(
            id="test-id", memory={"content": "Test content"}, user_id="test_user", last_updated=None
        )

        episode = memory_db._memory_to_episode(memory_without_timestamp)

        # Timestamp should be generated
        assert memory_without_timestamp.last_updated is not None
        assert episode["reference_time"] == memory_without_timestamp.last_updated

    def test_episode_to_memory_conversion(self, memory_db, sample_episode):
        """Test converting Graphiti episode to MemoryRow format."""
        memory = memory_db._episode_to_memory(sample_episode)

        assert memory.id == "test-memory-123"
        assert memory.memory == {"content": "User likes pizza", "topics": ["food", "preferences"]}
        assert memory.user_id == "test_user"
        assert memory.last_updated == datetime(2024, 1, 1, 12, 0, 0)

    def test_episode_to_memory_with_invalid_json(self, memory_db):
        """Test episode conversion with invalid JSON in episode_body."""
        invalid_episode = {
            "name": "memory_test-id",
            "episode_body": "invalid json {",
            "source_description": "agno_memory_test_memories",
            "group_id": "agno_test_memories_user_test_user",
        }

        memory = memory_db._episode_to_memory(invalid_episode)

        # Should fall back to wrapping content
        assert memory.memory == {"content": "invalid json {"}

    def test_episode_to_memory_with_string_timestamp(self, memory_db):
        """Test episode conversion with string timestamp."""
        episode_with_string_time = {
            "name": "memory_test-id",
            "episode_body": json.dumps({"content": "test"}),
            "reference_time": "2024-01-01T12:00:00Z",
            "group_id": "agno_test_memories_user_test_user",
        }

        memory = memory_db._episode_to_memory(episode_with_string_time)

        # Should parse the timestamp
        assert memory.last_updated is not None


class TestGroupIdMethods:
    """Test group ID generation and extraction."""

    def test_generate_group_id(self, memory_db):
        """Test group ID generation."""
        group_id = memory_db._generate_group_id("test_user")
        assert group_id == "agno_test_memories_user_test_user"

    def test_extract_user_id_success(self, memory_db):
        """Test successful user ID extraction."""
        group_id = "agno_test_memories_user_test_user"
        user_id = memory_db._extract_user_id(group_id)
        assert user_id == "test_user"

    def test_extract_user_id_failure(self, memory_db):
        """Test user ID extraction with invalid format."""
        invalid_group_id = "invalid_format"
        user_id = memory_db._extract_user_id(invalid_group_id)
        assert user_id is None


class TestMemoryExists:
    """Test memory existence checking."""

    def test_memory_exists_true(self, memory_db, mock_graphiti_client, sample_memory_row):
        """Test memory exists returns True when memory is found."""
        mock_graphiti_client.retrieve_episodes.return_value = [{"name": "memory_test-memory-123"}]

        exists = memory_db.memory_exists(sample_memory_row)

        assert exists is True
        mock_graphiti_client.retrieve_episodes.assert_called_once_with(episode_names=["memory_test-memory-123"])

    def test_memory_exists_false(self, memory_db, mock_graphiti_client, sample_memory_row):
        """Test memory exists returns False when memory is not found."""
        mock_graphiti_client.retrieve_episodes.return_value = []

        exists = memory_db.memory_exists(sample_memory_row)

        assert exists is False

    def test_memory_exists_no_id(self, memory_db):
        """Test memory exists returns False for memory without ID."""
        memory_without_id = MemoryRow(memory={"content": "test"}, user_id="test_user")

        exists = memory_db.memory_exists(memory_without_id)

        assert exists is False

    def test_memory_exists_exception(self, memory_db, mock_graphiti_client, sample_memory_row):
        """Test memory exists returns False on exception."""
        mock_graphiti_client.retrieve_episodes.side_effect = Exception("Connection error")

        exists = memory_db.memory_exists(sample_memory_row)

        assert exists is False


class TestQueryBuilders:
    """Test query builder utilities."""

    def test_build_episode_search_query_basic(self, memory_db):
        """Test basic search query building."""
        params = memory_db._build_episode_search_query()

        assert params["query"] == ""

    def test_build_episode_search_query_with_user_filter(self, memory_db):
        """Test search query building with user filter."""
        params = memory_db._build_episode_search_query(user_id="test_user")

        assert params["group_ids"] == ["agno_test_memories_user_test_user"]

    def test_build_episode_search_query_with_memory_ids(self, memory_db):
        """Test search query building with specific memory IDs."""
        params = memory_db._build_episode_search_query(memory_ids=["id1", "id2"])

        assert params["episode_names"] == ["memory_id1", "memory_id2"]

    def test_build_episode_search_query_with_content_query(self, memory_db):
        """Test search query building with content query."""
        params = memory_db._build_episode_search_query(content_query="pizza preferences")

        assert params["query"] == "pizza preferences"

    def test_build_episode_search_query_with_limit(self, memory_db):
        """Test search query building with limit."""
        params = memory_db._build_episode_search_query(limit=10)

        assert params["num_results"] == 10

    def test_optimize_search_parameters(self, memory_db):
        """Test search parameter optimization."""
        params = {"query": "test"}
        optimized = memory_db._optimize_search_parameters(params)

        # Should add default limit
        assert optimized["num_results"] == 1000
        assert optimized["_table_context"] == "test_memories"

    def test_optimize_search_parameters_with_high_limit(self, memory_db):
        """Test search parameter optimization with high limit."""
        params = {"num_results": 5000}
        optimized = memory_db._optimize_search_parameters(params)

        # Should cap at 1000
        assert optimized["num_results"] == 1000

    def test_validate_episode_belongs_to_table(self, memory_db, sample_episode):
        """Test episode table validation."""
        assert memory_db._validate_episode_belongs_to_table(sample_episode) is True

    def test_validate_episode_wrong_table(self, memory_db):
        """Test episode table validation for wrong table."""
        wrong_episode = {"source_description": "agno_memory_other_table"}
        assert memory_db._validate_episode_belongs_to_table(wrong_episode) is False


class TestUpsertMemory:
    """Test memory upsert operations."""

    def test_upsert_new_memory(self, memory_db, mock_graphiti_client, sample_memory_row):
        """Test upserting a new memory."""
        mock_graphiti_client.retrieve_episodes.return_value = []  # Memory doesn't exist

        result = memory_db.upsert_memory(sample_memory_row)

        # Should add episode without removing first
        mock_graphiti_client.add_episode.assert_called_once()
        mock_graphiti_client.remove_episode.assert_not_called()
        assert result == sample_memory_row

    def test_upsert_existing_memory(self, memory_db, mock_graphiti_client, sample_memory_row):
        """Test upserting an existing memory."""
        mock_graphiti_client.retrieve_episodes.return_value = [sample_memory_row]  # Memory exists

        result = memory_db.upsert_memory(sample_memory_row)

        # Should remove existing and add new
        mock_graphiti_client.remove_episode.assert_called_once_with("memory_test-memory-123")
        mock_graphiti_client.add_episode.assert_called_once()
        assert result == sample_memory_row

    def test_upsert_memory_failure(self, memory_db, mock_graphiti_client, sample_memory_row):
        """Test upsert memory failure handling."""
        mock_graphiti_client.add_episode.side_effect = Exception("Add failed")

        with pytest.raises(RuntimeError, match="Failed to upsert memory"):
            memory_db.upsert_memory(sample_memory_row)


class TestReadMemories:
    """Test memory reading operations."""

    def test_read_memories_empty(self, memory_db, mock_graphiti_client):
        """Test reading memories when none exist."""
        mock_graphiti_client.search.return_value = []

        memories = memory_db.read_memories()

        assert memories == []

    def test_read_memories_with_user_filter(self, memory_db, mock_graphiti_client, sample_episode):
        """Test reading memories with user filter."""
        mock_graphiti_client.search.return_value = [sample_episode]

        memories = memory_db.read_memories(user_id="test_user")

        # Verify search was called with user filter
        mock_graphiti_client.search.assert_called_once()
        call_args = mock_graphiti_client.search.call_args
        assert "group_ids" in call_args.kwargs
        assert call_args.kwargs["group_ids"] == ["agno_test_memories_user_test_user"]

        assert len(memories) == 1
        assert memories[0].id == "test-memory-123"

    def test_read_memories_with_limit(self, memory_db, mock_graphiti_client, sample_episode):
        """Test reading memories with limit."""
        mock_graphiti_client.search.return_value = [sample_episode]

        memory_db.read_memories(limit=5)

        # Verify limit was applied
        call_args = mock_graphiti_client.search.call_args
        assert call_args.kwargs["num_results"] == 5

    def test_read_memories_with_sorting_asc(self, memory_db, mock_graphiti_client):
        """Test reading memories with ascending sort."""
        episode1 = {
            "name": "memory_1",
            "episode_body": json.dumps({"content": "First"}),
            "source_description": "agno_memory_test_memories",
            "reference_time": datetime(2024, 1, 1),
            "group_id": "agno_test_memories_user_test_user",
        }
        episode2 = {
            "name": "memory_2",
            "episode_body": json.dumps({"content": "Second"}),
            "source_description": "agno_memory_test_memories",
            "reference_time": datetime(2024, 1, 2),
            "group_id": "agno_test_memories_user_test_user",
        }

        mock_graphiti_client.search.return_value = [episode2, episode1]  # Reverse order

        memories = memory_db.read_memories(sort="asc")

        # Should be sorted by timestamp ascending
        assert len(memories) == 2
        assert memories[0].last_updated < memories[1].last_updated

    def test_read_memories_filters_wrong_table(self, memory_db, mock_graphiti_client):
        """Test that memories from wrong table are filtered out."""
        wrong_episode = {"name": "memory_wrong", "source_description": "agno_memory_other_table"}

        mock_graphiti_client.search.return_value = [wrong_episode]

        memories = memory_db.read_memories()

        # Should filter out episodes from other tables
        assert memories == []

    def test_read_memories_handles_conversion_errors(self, memory_db, mock_graphiti_client):
        """Test that conversion errors are handled gracefully."""
        invalid_episode = {
            "name": "invalid",
            # Missing required fields to cause conversion error
        }

        mock_graphiti_client.search.return_value = [invalid_episode]

        memories = memory_db.read_memories()

        # Should skip invalid episodes
        assert memories == []

    def test_read_memories_failure(self, memory_db, mock_graphiti_client):
        """Test read memories failure handling."""
        mock_graphiti_client.search.side_effect = Exception("Search failed")

        with pytest.raises(RuntimeError, match="Failed to read memories"):
            memory_db.read_memories()


class TestDeleteMemory:
    """Test memory deletion operations."""

    def test_delete_memory_success(self, memory_db, mock_graphiti_client):
        """Test successful memory deletion."""
        memory_db.delete_memory("test-memory-123")

        mock_graphiti_client.remove_episode.assert_called_once_with("memory_test-memory-123")

    def test_delete_memory_failure(self, memory_db, mock_graphiti_client):
        """Test memory deletion failure handling."""
        mock_graphiti_client.remove_episode.side_effect = Exception("Delete failed")

        with pytest.raises(RuntimeError, match="Failed to delete memory test-memory-123"):
            memory_db.delete_memory("test-memory-123")


class TestTableOperations:
    """Test table-level operations."""

    def test_table_exists_true(self, memory_db, mock_graphiti_client, sample_episode):
        """Test table exists returns True when memories exist."""
        mock_graphiti_client.search.return_value = [sample_episode]

        exists = memory_db.table_exists()

        assert exists is True

    def test_table_exists_false(self, memory_db, mock_graphiti_client):
        """Test table exists returns False when no memories exist."""
        mock_graphiti_client.search.return_value = []

        exists = memory_db.table_exists()

        assert exists is False

    def test_table_exists_exception(self, memory_db, mock_graphiti_client):
        """Test table exists returns False on exception."""
        mock_graphiti_client.search.side_effect = Exception("Search failed")

        exists = memory_db.table_exists()

        assert exists is False

    def test_drop_table(self, memory_db, mock_graphiti_client, sample_episode):
        """Test dropping table (deleting all memories)."""
        mock_graphiti_client.search.return_value = [sample_episode]

        memory_db.drop_table()

        # Should call remove_episode for each memory found
        mock_graphiti_client.remove_episode.assert_called_once_with("memory_test-memory-123")

    def test_drop_table_failure(self, memory_db, mock_graphiti_client):
        """Test drop table failure handling."""
        mock_graphiti_client.search.side_effect = Exception("Search failed")

        with pytest.raises(RuntimeError, match="Failed to drop table test_memories"):
            memory_db.drop_table()

    def test_clear_success(self, memory_db, mock_graphiti_client, sample_episode):
        """Test successful table clearing."""
        mock_graphiti_client.search.return_value = [sample_episode]

        result = memory_db.clear()

        assert result is True
        mock_graphiti_client.remove_episode.assert_called_once()

    def test_clear_failure(self, memory_db, mock_graphiti_client):
        """Test table clearing failure handling."""
        mock_graphiti_client.search.side_effect = Exception("Clear failed")

        result = memory_db.clear()

        assert result is False


class TestSimilaritySearch:
    """Test similarity search functionality."""

    def test_similarity_search_success(self, memory_db, mock_graphiti_client, sample_episode):
        """Test successful similarity search."""
        mock_graphiti_client.search.return_value = [sample_episode]

        memories = memory_db.similarity_search("pizza preferences")

        # Verify search was called with query
        call_args = mock_graphiti_client.search.call_args
        assert call_args.kwargs["query"] == "pizza preferences"

        assert len(memories) == 1
        assert memories[0].id == "test-memory-123"

    def test_similarity_search_with_user_filter(self, memory_db, mock_graphiti_client, sample_episode):
        """Test similarity search with user filter."""
        mock_graphiti_client.search.return_value = [sample_episode]

        memory_db.similarity_search("pizza", user_id="test_user", limit=5)

        call_args = mock_graphiti_client.search.call_args
        assert call_args.kwargs["query"] == "pizza"
        assert call_args.kwargs["group_ids"] == ["agno_test_memories_user_test_user"]
        assert call_args.kwargs["num_results"] == 5

    def test_similarity_search_failure(self, memory_db, mock_graphiti_client):
        """Test similarity search failure handling."""
        mock_graphiti_client.search.side_effect = Exception("Search failed")

        with pytest.raises(RuntimeError, match="Failed to perform similarity search"):
            memory_db.similarity_search("test query")


class TestBulkOperations:
    """Test bulk memory operations."""

    def test_bulk_upsert_memories_success(self, memory_db, mock_graphiti_client):
        """Test successful bulk upsert of memories."""
        memories = [
            MemoryRow(id="1", memory={"content": "First"}, user_id="user1"),
            MemoryRow(id="2", memory={"content": "Second"}, user_id="user2"),
        ]

        mock_graphiti_client.retrieve_episodes.return_value = []  # Memories don't exist

        result = memory_db.bulk_upsert_memories(memories)

        assert len(result) == 2
        assert mock_graphiti_client.add_episode.call_count == 2

    def test_bulk_upsert_memories_partial_failure(self, memory_db, mock_graphiti_client):
        """Test bulk upsert with partial failures."""
        memories = [
            MemoryRow(id="1", memory={"content": "First"}, user_id="user1"),
            MemoryRow(id="2", memory={"content": "Second"}, user_id="user2"),
        ]

        mock_graphiti_client.retrieve_episodes.return_value = []
        # Make second add_episode call fail
        mock_graphiti_client.add_episode.side_effect = [None, Exception("Failed"), None]

        result = memory_db.bulk_upsert_memories(memories)

        # Should return only successful memories
        assert len(result) == 1

    def test_bulk_delete_memories_success(self, memory_db, mock_graphiti_client):
        """Test successful bulk deletion of memories."""
        memory_ids = ["id1", "id2", "id3"]

        result = memory_db.bulk_delete_memories(memory_ids)

        assert result == 3
        assert mock_graphiti_client.remove_episode.call_count == 3

    def test_bulk_delete_memories_partial_failure(self, memory_db, mock_graphiti_client):
        """Test bulk deletion with partial failures."""
        memory_ids = ["id1", "id2", "id3"]

        # Make second delete fail
        mock_graphiti_client.remove_episode.side_effect = [None, Exception("Failed"), None]

        result = memory_db.bulk_delete_memories(memory_ids)

        # Should return count of successful deletions
        assert result == 2

    def test_build_bulk_operation_params_upsert(self, memory_db):
        """Test building bulk operation parameters for upsert."""
        memories = [
            MemoryRow(id="1", memory={"content": "First"}, user_id="user1"),
            MemoryRow(id="2", memory={"content": "Second"}, user_id="user2"),
        ]

        params = memory_db._build_bulk_operation_params(memories, "upsert")

        assert len(params) == 2
        assert params[0]["name"] == "memory_1"
        assert params[1]["name"] == "memory_2"

    def test_build_bulk_operation_params_delete(self, memory_db):
        """Test building bulk operation parameters for delete."""
        memories = [
            MemoryRow(id="1", memory={"content": "First"}, user_id="user1"),
            MemoryRow(id="2", memory={"content": "Second"}, user_id="user2"),
        ]

        params = memory_db._build_bulk_operation_params(memories, "delete")

        assert len(params) == 2
        assert params[0]["episode_name"] == "memory_1"
        assert params[1]["episode_name"] == "memory_2"

    def test_build_bulk_operation_params_invalid_operation(self, memory_db):
        """Test building bulk operation parameters with invalid operation."""
        memories = [MemoryRow(memory={"content": "test"})]

        with pytest.raises(ValueError, match="Unsupported operation type"):
            memory_db._build_bulk_operation_params(memories, "invalid")


class TestFilterTranslation:
    """Test filter translation functionality."""

    def test_translate_filters_empty(self, memory_db):
        """Test filter translation with no filters."""
        filters = memory_db._translate_filters()

        assert filters == {}

    def test_translate_filters_user_id(self, memory_db):
        """Test filter translation with user_id."""
        filters = memory_db._translate_filters(user_id="test_user")

        assert filters["group_ids"] == ["agno_test_memories_user_test_user"]

    def test_translate_filters_limit(self, memory_db):
        """Test filter translation with limit."""
        filters = memory_db._translate_filters(limit=10)

        assert filters["num_results"] == 10

    def test_translate_filters_combined(self, memory_db):
        """Test filter translation with multiple filters."""
        filters = memory_db._translate_filters(user_id="test_user", limit=5)

        assert filters["group_ids"] == ["agno_test_memories_user_test_user"]
        assert filters["num_results"] == 5


# Integration-style tests
class TestEndToEndScenarios:
    """Test complete workflows."""

    def test_complete_memory_lifecycle(self, memory_db, mock_graphiti_client):
        """Test complete memory lifecycle: create, read, update, delete."""
        # Create memory
        memory = MemoryRow(memory={"content": "Test memory", "topics": ["test"]}, user_id="test_user")

        mock_graphiti_client.retrieve_episodes.return_value = []  # Doesn't exist initially

        # Upsert memory
        result = memory_db.upsert_memory(memory)
        assert result == memory

        # Read memories
        mock_graphiti_client.search.return_value = [
            {
                "name": f"memory_{memory.id}",
                "episode_body": json.dumps(memory.memory),
                "source_description": "agno_memory_test_memories",
                "reference_time": memory.last_updated,
                "group_id": "agno_test_memories_user_test_user",
            }
        ]

        memories = memory_db.read_memories(user_id="test_user")
        assert len(memories) == 1
        assert memories[0].memory["content"] == "Test memory"

        # Delete memory
        memory_db.delete_memory(memory.id)
        mock_graphiti_client.remove_episode.assert_called_with(f"memory_{memory.id}")

    def test_multi_user_isolation(self, memory_db, mock_graphiti_client):
        """Test that memories from different users are properly isolated."""
        user1_episode = {
            "name": "memory_user1_mem",
            "episode_body": json.dumps({"content": "User 1 memory"}),
            "source_description": "agno_memory_test_memories",
            "group_id": "agno_test_memories_user_user1",
        }

        # Mock search to return user1 episode when filtering by user1
        mock_graphiti_client.search.return_value = [user1_episode]

        memory_db.read_memories(user_id="user1")

        # Verify search was called with correct group filter
        call_args = mock_graphiti_client.search.call_args
        assert call_args.kwargs["group_ids"] == ["agno_test_memories_user_user1"]
