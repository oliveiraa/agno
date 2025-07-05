"""
Contract tests for MemoryDb implementations.

These tests ensure all MemoryDb implementations comply with the base interface contract.
They can be run against any implementation by providing the appropriate setup fixture.
"""

import uuid
from datetime import datetime
from typing import List

import pytest

from agno.memory.v2.db.base import MemoryDb
from agno.memory.v2.db.schema import MemoryRow


class MemoryDbContractTests:
    """
    Base contract test class for MemoryDb implementations.

    Subclass this and provide a memory_db fixture to test your implementation.
    """

    @pytest.fixture
    def memory_db(self) -> MemoryDb:
        """Override this fixture to provide your MemoryDb implementation."""
        raise NotImplementedError("Subclasses must provide a memory_db fixture")

    @pytest.fixture
    def sample_memory(self) -> MemoryRow:
        """Create a sample memory for testing."""
        return MemoryRow(
            id=str(uuid.uuid4()),
            memory={"content": "Test memory content", "tags": ["test", "sample"]},
            user_id="test_user",
            last_updated=datetime.now(),
        )

    @pytest.fixture
    def sample_memories(self) -> List[MemoryRow]:
        """Create multiple sample memories for testing."""
        memories = []
        for i in range(5):
            memory = MemoryRow(
                id=str(uuid.uuid4()),
                memory={"content": f"Test memory {i}", "index": i},
                user_id=f"user_{i % 2}",  # Two different users
                last_updated=datetime.now(),
            )
            memories.append(memory)
        return memories

    def test_create(self, memory_db: MemoryDb):
        """Test database initialization."""
        # Should not raise any exceptions
        memory_db.create()

    def test_table_operations(self, memory_db: MemoryDb):
        """Test table existence and drop operations."""
        memory_db.create()

        # After creation, table should exist
        assert memory_db.table_exists() is True

        # Drop the table
        memory_db.drop_table()

        # After dropping, table should not exist
        assert memory_db.table_exists() is False

    def test_memory_lifecycle(self, memory_db: MemoryDb, sample_memory: MemoryRow):
        """Test the complete lifecycle of a memory: create, read, update, delete."""
        memory_db.create()

        # Initially, memory should not exist
        assert memory_db.memory_exists(sample_memory) is False

        # Upsert the memory
        result = memory_db.upsert_memory(sample_memory)
        assert result is not None
        assert result.id == sample_memory.id

        # Now memory should exist
        assert memory_db.memory_exists(sample_memory) is True

        # Read memories should include our memory
        memories = memory_db.read_memories()
        assert len(memories) == 1
        assert memories[0].id == sample_memory.id
        assert memories[0].memory == sample_memory.memory

        # Update the memory
        sample_memory.memory["content"] = "Updated content"
        memory_db.upsert_memory(sample_memory)

        # Read again and verify update
        memories = memory_db.read_memories()
        assert len(memories) == 1
        assert memories[0].memory["content"] == "Updated content"

        # Delete the memory
        memory_db.delete_memory(sample_memory.id)

        # Memory should no longer exist
        assert memory_db.memory_exists(sample_memory) is False
        memories = memory_db.read_memories()
        assert len(memories) == 0

    def test_read_memories_filtering(self, memory_db: MemoryDb, sample_memories: List[MemoryRow]):
        """Test reading memories with various filters."""
        memory_db.create()

        # Insert all sample memories
        for memory in sample_memories:
            memory_db.upsert_memory(memory)

        # Read all memories
        all_memories = memory_db.read_memories()
        assert len(all_memories) == len(sample_memories)

        # Read with user filter
        user_0_memories = memory_db.read_memories(user_id="user_0")
        assert all(m.user_id == "user_0" for m in user_0_memories)
        assert len(user_0_memories) == 3  # 0, 2, 4

        user_1_memories = memory_db.read_memories(user_id="user_1")
        assert all(m.user_id == "user_1" for m in user_1_memories)
        assert len(user_1_memories) == 2  # 1, 3

        # Read with limit
        limited_memories = memory_db.read_memories(limit=2)
        assert len(limited_memories) == 2

        # Read with user filter and limit
        limited_user_memories = memory_db.read_memories(user_id="user_0", limit=1)
        assert len(limited_user_memories) == 1
        assert limited_user_memories[0].user_id == "user_0"

    def test_read_memories_sorting(self, memory_db: MemoryDb, sample_memories: List[MemoryRow]):
        """Test reading memories with sort order."""
        memory_db.create()

        # Insert memories in reverse order
        for memory in reversed(sample_memories):
            memory_db.upsert_memory(memory)

        # Read with ascending sort
        asc_memories = memory_db.read_memories(sort="asc")
        # Verify they're sorted by last_updated ascending
        for i in range(1, len(asc_memories)):
            assert asc_memories[i - 1].last_updated <= asc_memories[i].last_updated

        # Read with descending sort (default)
        desc_memories = memory_db.read_memories(sort="desc")
        # Verify they're sorted by last_updated descending
        for i in range(1, len(desc_memories)):
            assert desc_memories[i - 1].last_updated >= desc_memories[i].last_updated

    def test_clear(self, memory_db: MemoryDb, sample_memories: List[MemoryRow]):
        """Test clearing all memories."""
        memory_db.create()

        # Insert some memories
        for memory in sample_memories:
            memory_db.upsert_memory(memory)

        # Verify they exist
        assert len(memory_db.read_memories()) == len(sample_memories)

        # Clear the database
        result = memory_db.clear()
        assert result is True

        # Verify all memories are gone
        assert len(memory_db.read_memories()) == 0

    def test_memory_without_id(self, memory_db: MemoryDb):
        """Test upserting a memory without an ID (auto-generation)."""
        memory_db.create()

        # Create memory without ID
        memory = MemoryRow(memory={"content": "No ID memory"}, user_id="test_user")

        # Upsert should auto-generate ID
        result = memory_db.upsert_memory(memory)
        assert result is not None
        assert result.id is not None
        assert len(result.id) > 0

        # Memory should be retrievable
        memories = memory_db.read_memories()
        assert len(memories) == 1
        assert memories[0].id == result.id

    def test_memory_without_timestamp(self, memory_db: MemoryDb):
        """Test upserting a memory without a timestamp (auto-generation)."""
        memory_db.create()

        # Create memory without timestamp
        memory = MemoryRow(id=str(uuid.uuid4()), memory={"content": "No timestamp memory"}, user_id="test_user")

        # Upsert should auto-generate timestamp
        result = memory_db.upsert_memory(memory)
        assert result is not None
        assert result.last_updated is not None

        # Memory should be retrievable with timestamp
        memories = memory_db.read_memories()
        assert len(memories) == 1
        assert memories[0].last_updated is not None
