"""
Integration tests for GraphitiMemoryDb with real Neo4j database.

These tests run against a real Neo4j instance to validate the complete
integration works correctly with actual graph database operations.
"""

import os
import time
from datetime import datetime
from typing import List

import pytest

from agno.memory.v2.db.graphiti import GraphitiMemoryDb
from agno.memory.v2.db.schema import MemoryRow

# Test configuration
NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "dummy-key-for-testing")


@pytest.fixture(scope="session")
def neo4j_ready():
    """Ensure Neo4j is ready before running tests."""
    max_retries = 30
    retry_delay = 1

    for attempt in range(max_retries):
        try:
            # Try to create a simple GraphitiMemoryDb instance
            db = GraphitiMemoryDb(uri=NEO4J_URI, user=NEO4J_USER, password=NEO4J_PASSWORD, table_name="connection_test")
            db.create()
            return True
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
            pytest.skip(f"Neo4j not available after {max_retries} attempts: {e}")

    pytest.skip("Neo4j not available")


@pytest.fixture
def memory_db(neo4j_ready):
    """Create a GraphitiMemoryDb instance for testing."""
    # Set dummy OpenAI API key if not set
    if not os.getenv("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

    db = GraphitiMemoryDb(uri=NEO4J_URI, user=NEO4J_USER, password=NEO4J_PASSWORD, table_name="integration_test")

    # Initialize database
    db.create()

    # Clean up any existing test data
    db.clear()

    yield db

    # Cleanup after test
    try:
        db.clear()
    except Exception:
        pass  # Ignore cleanup errors


@pytest.fixture
def sample_memories() -> List[MemoryRow]:
    """Create sample memory data for testing."""
    return [
        MemoryRow(
            memory={"content": "User likes pizza", "topics": ["food", "preferences"]},
            user_id="user1",
            last_updated=datetime(2024, 1, 1, 12, 0, 0),
        ),
        MemoryRow(
            memory={"content": "User works as a software engineer", "topics": ["work", "profession"]},
            user_id="user1",
            last_updated=datetime(2024, 1, 2, 12, 0, 0),
        ),
        MemoryRow(
            memory={"content": "User enjoys hiking on weekends", "topics": ["hobbies", "outdoor"]},
            user_id="user2",
            last_updated=datetime(2024, 1, 3, 12, 0, 0),
        ),
    ]


class TestGraphitiMemoryDbIntegration:
    """Integration tests for GraphitiMemoryDb."""

    def test_database_initialization(self, memory_db):
        """Test that database can be initialized successfully."""
        # Should not raise any exceptions
        memory_db.create()

        # Table should not exist initially (no memories)
        assert memory_db.table_exists() is False

    def test_single_memory_lifecycle(self, memory_db):
        """Test complete lifecycle of a single memory."""
        # Create memory
        memory = MemoryRow(memory={"content": "Integration test memory", "topics": ["test"]}, user_id="test_user")

        # Initially memory should not exist
        assert memory_db.memory_exists(memory) is False

        # Upsert memory
        result = memory_db.upsert_memory(memory)
        assert result is not None
        assert result.id is not None

        # Memory should now exist
        assert memory_db.memory_exists(memory) is True

        # Table should exist
        assert memory_db.table_exists() is True

        # Read memories
        memories = memory_db.read_memories()
        assert len(memories) == 1
        assert memories[0].memory["content"] == "Integration test memory"
        assert memories[0].user_id == "test_user"

        # Update memory
        memory.memory["content"] = "Updated integration test memory"
        updated_result = memory_db.upsert_memory(memory)
        assert updated_result is not None

        # Verify update
        updated_memories = memory_db.read_memories()
        assert len(updated_memories) == 1
        assert updated_memories[0].memory["content"] == "Updated integration test memory"

        # Delete memory
        memory_db.delete_memory(memory.id)

        # Memory should no longer exist
        assert memory_db.memory_exists(memory) is False

        # Should have no memories
        memories = memory_db.read_memories()
        assert len(memories) == 0

    def test_multiple_memories_operations(self, memory_db, sample_memories):
        """Test operations with multiple memories."""
        # Upsert all memories
        for memory in sample_memories:
            result = memory_db.upsert_memory(memory)
            assert result is not None

        # Read all memories
        all_memories = memory_db.read_memories()
        assert len(all_memories) == 3

        # Test user filtering
        user1_memories = memory_db.read_memories(user_id="user1")
        assert len(user1_memories) == 2
        assert all(m.user_id == "user1" for m in user1_memories)

        user2_memories = memory_db.read_memories(user_id="user2")
        assert len(user2_memories) == 1
        assert user2_memories[0].user_id == "user2"

        # Test limit
        limited_memories = memory_db.read_memories(limit=2)
        assert len(limited_memories) == 2

        # Test sorting
        asc_memories = memory_db.read_memories(sort="asc")
        desc_memories = memory_db.read_memories(sort="desc")

        # Should be in opposite order
        assert asc_memories[0].last_updated <= asc_memories[-1].last_updated
        assert desc_memories[0].last_updated >= desc_memories[-1].last_updated

    def test_similarity_search(self, memory_db, sample_memories):
        """Test vector similarity search functionality."""
        # Add sample memories
        for memory in sample_memories:
            memory_db.upsert_memory(memory)

        # Wait a moment for indexing
        time.sleep(2)

        # Search for food-related content
        food_memories = memory_db.similarity_search("food and eating")

        # Should find the pizza memory
        assert len(food_memories) > 0
        food_contents = [m.memory.get("content", "") for m in food_memories]
        assert any("pizza" in content.lower() for content in food_contents)

        # Search with user filter
        user1_food_memories = memory_db.similarity_search("food", user_id="user1")
        assert all(m.user_id == "user1" for m in user1_food_memories)

        # Search with limit
        limited_search = memory_db.similarity_search("user", limit=1)
        assert len(limited_search) <= 1

    def test_bulk_operations(self, memory_db, sample_memories):
        """Test bulk memory operations."""
        # Bulk upsert
        results = memory_db.bulk_upsert_memories(sample_memories)
        assert len(results) == len(sample_memories)

        # Verify all were inserted
        all_memories = memory_db.read_memories()
        assert len(all_memories) == len(sample_memories)

        # Bulk delete
        memory_ids = [m.id for m in results if m.id]
        deleted_count = memory_db.bulk_delete_memories(memory_ids)
        assert deleted_count == len(memory_ids)

        # Verify all were deleted
        remaining_memories = memory_db.read_memories()
        assert len(remaining_memories) == 0

    def test_table_operations(self, memory_db, sample_memories):
        """Test table-level operations."""
        # Initially table should not exist
        assert memory_db.table_exists() is False

        # Add some memories
        for memory in sample_memories:
            memory_db.upsert_memory(memory)

        # Table should now exist
        assert memory_db.table_exists() is True

        # Clear table
        assert memory_db.clear() is True

        # Table should still exist but be empty
        memories = memory_db.read_memories()
        assert len(memories) == 0

        # Add memories again
        for memory in sample_memories:
            memory_db.upsert_memory(memory)

        # Drop table
        memory_db.drop_table()

        # Should have no memories
        memories = memory_db.read_memories()
        assert len(memories) == 0

    def test_concurrent_operations(self, memory_db):
        """Test concurrent operations on the same memory."""
        memory = MemoryRow(memory={"content": "Concurrent test", "topics": ["concurrency"]}, user_id="concurrent_user")

        # Multiple upserts should work
        for i in range(5):
            memory.memory["content"] = f"Concurrent test {i}"
            result = memory_db.upsert_memory(memory)
            assert result is not None

        # Should have only one memory
        memories = memory_db.read_memories(user_id="concurrent_user")
        assert len(memories) == 1
        assert "Concurrent test 4" in memories[0].memory["content"]

    def test_error_handling(self, memory_db):
        """Test error handling with invalid operations."""
        # Try to delete non-existent memory (should not raise)
        memory_db.delete_memory("non-existent-id")

        # Try to read with invalid user (should return empty)
        memories = memory_db.read_memories(user_id="non-existent-user")
        assert memories == []

        # Try similarity search with empty query (should not raise)
        results = memory_db.similarity_search("")
        assert isinstance(results, list)

    def test_memory_with_auto_generated_fields(self, memory_db):
        """Test memory with auto-generated ID and timestamp."""
        memory = MemoryRow(memory={"content": "Auto-generated test"}, user_id="auto_user")

        # ID and timestamp should be None initially
        assert memory.id is None
        assert memory.last_updated is None

        # Upsert should generate them
        result = memory_db.upsert_memory(memory)

        # Should now have ID and timestamp
        assert memory.id is not None
        assert memory.last_updated is not None
        assert result.id == memory.id

    def test_user_isolation(self, memory_db):
        """Test that memories from different users are properly isolated."""
        # Create memories for different users
        user1_memory = MemoryRow(memory={"content": "User 1 secret", "topics": ["secret"]}, user_id="user1")

        user2_memory = MemoryRow(memory={"content": "User 2 secret", "topics": ["secret"]}, user_id="user2")

        memory_db.upsert_memory(user1_memory)
        memory_db.upsert_memory(user2_memory)

        # Each user should only see their own memories
        user1_memories = memory_db.read_memories(user_id="user1")
        user2_memories = memory_db.read_memories(user_id="user2")

        assert len(user1_memories) == 1
        assert len(user2_memories) == 1
        assert user1_memories[0].memory["content"] == "User 1 secret"
        assert user2_memories[0].memory["content"] == "User 2 secret"

        # Similarity search should also respect user isolation
        user1_search = memory_db.similarity_search("secret", user_id="user1")
        assert len(user1_search) >= 1
        assert all(m.user_id == "user1" for m in user1_search)

    def test_complex_memory_data(self, memory_db):
        """Test with complex nested memory data structures."""
        complex_memory = MemoryRow(
            memory={
                "content": "Complex data structure",
                "topics": ["data", "structures", "complex"],
                "metadata": {"source": "integration_test", "confidence": 0.95, "tags": ["important", "verified"]},
                "relationships": [
                    {"type": "relates_to", "target": "memory_123"},
                    {"type": "derived_from", "target": "document_456"},
                ],
            },
            user_id="complex_user",
        )

        # Upsert complex memory
        result = memory_db.upsert_memory(complex_memory)
        assert result is not None

        # Read back and verify structure
        memories = memory_db.read_memories(user_id="complex_user")
        assert len(memories) == 1

        retrieved_memory = memories[0]
        assert retrieved_memory.memory["content"] == "Complex data structure"
        assert "metadata" in retrieved_memory.memory
        assert retrieved_memory.memory["metadata"]["confidence"] == 0.95
        assert len(retrieved_memory.memory["relationships"]) == 2


@pytest.mark.slow
class TestGraphitiMemoryDbPerformance:
    """Performance-focused integration tests."""

    def test_batch_performance(self, memory_db):
        """Test performance with larger batches of memories."""
        # Create 100 memories
        memories = []
        for i in range(100):
            memory = MemoryRow(
                memory={
                    "content": f"Performance test memory {i}",
                    "topics": ["performance", "test", f"batch_{i // 10}"],
                },
                user_id=f"perf_user_{i % 5}",  # 5 different users
            )
            memories.append(memory)

        # Time bulk upsert
        start_time = time.time()
        results = memory_db.bulk_upsert_memories(memories)
        upsert_time = time.time() - start_time

        assert len(results) == 100
        print(f"Bulk upsert of 100 memories took {upsert_time:.2f} seconds")

        # Time read all
        start_time = time.time()
        all_memories = memory_db.read_memories()
        read_time = time.time() - start_time

        assert len(all_memories) == 100
        print(f"Reading 100 memories took {read_time:.2f} seconds")

        # Time similarity search
        start_time = time.time()
        search_results = memory_db.similarity_search("performance test")
        search_time = time.time() - start_time

        print(f"Similarity search took {search_time:.2f} seconds, found {len(search_results)} results")

        # Cleanup
        memory_ids = [m.id for m in results if m.id]
        deleted_count = memory_db.bulk_delete_memories(memory_ids)
        assert deleted_count == len(memory_ids)


if __name__ == "__main__":
    # Can be run directly for manual testing
    pytest.main([__file__, "-v", "-s"])
