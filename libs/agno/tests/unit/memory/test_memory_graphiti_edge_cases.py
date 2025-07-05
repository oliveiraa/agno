"""
Additional edge case tests for GraphitiMemoryDb to improve test coverage.
"""

import json
import pytest
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch

from agno.memory.v2.db.graphiti import GraphitiMemoryDb
from agno.memory.v2.db.schema import MemoryRow


class TestEdgeCases:
    """Test edge cases and error conditions for better coverage."""

    def test_initialization_with_invalid_params(self):
        """Test initialization with invalid configuration parameters."""
        # Test invalid max_results
        with pytest.raises(ValueError, match="max_results must be positive"):
            GraphitiMemoryDb(
                uri="bolt://localhost:7687", 
                user="neo4j", 
                password="password",
                max_results=0
            )
        
        # Test invalid max_json_size_mb
        with pytest.raises(ValueError, match="max_json_size_mb must be positive"):
            GraphitiMemoryDb(
                uri="bolt://localhost:7687", 
                user="neo4j", 
                password="password",
                max_json_size_mb=-1
            )

    def test_generate_group_id_validation(self):
        """Test group ID generation with invalid user IDs."""
        mock_graphiti = Mock()
        with patch("agno.memory.v2.db.graphiti.Graphiti", return_value=mock_graphiti):
            db = GraphitiMemoryDb(uri="bolt://localhost:7687", user="neo4j", password="password")
            
            # Test empty user_id
            with pytest.raises(ValueError, match="user_id cannot be empty"):
                db._generate_group_id("")
            
            # Test invalid characters
            with pytest.raises(ValueError, match="Invalid user_id format"):
                db._generate_group_id("user@with#invalid$chars")
            
            # Test too long user_id
            with pytest.raises(ValueError, match="Invalid user_id format"):
                db._generate_group_id("a" * 65)

    def test_episode_to_memory_large_json(self):
        """Test episode conversion with large JSON that exceeds size limit."""
        mock_graphiti = Mock()
        with patch("agno.memory.v2.db.graphiti.Graphiti", return_value=mock_graphiti):
            db = GraphitiMemoryDb(
                uri="bolt://localhost:7687", 
                user="neo4j", 
                password="password",
                max_json_size_mb=1  # 1MB limit
            )
            
            # Create episode with large body
            large_content = "x" * (2 * 1024 * 1024)  # 2MB
            episode = {
                "name": "memory_123",
                "episode_body": json.dumps({"content": large_content}),
                "source_description": "agno_memory_agno_memories",
                "reference_time": datetime.now().isoformat()
            }
            
            # Should handle gracefully
            memory = db._episode_to_memory(episode)
            assert memory.memory == {"content": "INVALID_DATA"}

    def test_episode_to_memory_non_dict_json(self):
        """Test episode conversion when JSON deserializes to non-dict."""
        mock_graphiti = Mock()
        with patch("agno.memory.v2.db.graphiti.Graphiti", return_value=mock_graphiti):
            db = GraphitiMemoryDb(uri="bolt://localhost:7687", user="neo4j", password="password")
            
            episode = {
                "name": "memory_123",
                "episode_body": json.dumps(["not", "a", "dict"]),  # List instead of dict
                "source_description": "agno_memory_agno_memories",
                "reference_time": datetime.now().isoformat()
            }
            
            # Should handle gracefully
            memory = db._episode_to_memory(episode)
            assert memory.memory == {"content": "INVALID_DATA"}

    def test_build_episode_filter_criteria(self):
        """Test episode filter criteria building."""
        mock_graphiti = Mock()
        with patch("agno.memory.v2.db.graphiti.Graphiti", return_value=mock_graphiti):
            db = GraphitiMemoryDb(uri="bolt://localhost:7687", user="neo4j", password="password")
            
            # Test with table name filter
            criteria = db._build_episode_filter_criteria(table_name_filter=True)
            assert criteria["source_description_prefix"] == "agno_memory_agno_memories"
            
            # Test without table name filter
            criteria = db._build_episode_filter_criteria(table_name_filter=False)
            assert "source_description_prefix" not in criteria
            
            # Test with additional filters
            criteria = db._build_episode_filter_criteria(
                table_name_filter=True,
                additional_filters={"custom": "value"}
            )
            assert criteria["source_description_prefix"] == "agno_memory_agno_memories"
            assert criteria["custom"] == "value"

    def test_memory_without_user_id(self):
        """Test memory operations without user_id."""
        mock_graphiti = Mock()
        mock_graphiti.build_indices_for_schema = AsyncMock(return_value=None)
        mock_graphiti.add_episode = AsyncMock(return_value=None)
        
        with patch("agno.memory.v2.db.graphiti.Graphiti", return_value=mock_graphiti):
            db = GraphitiMemoryDb(uri="bolt://localhost:7687", user="neo4j", password="password")
            
            # Create memory without user_id
            memory = MemoryRow(
                memory={"content": "No user memory"}
            )
            
            # Should handle gracefully
            result = db.upsert_memory(memory)
            assert result.id is not None
            
            # Verify episode was created without group_id
            call_args = mock_graphiti.add_episode.call_args
            assert call_args.kwargs["group_id"] is None

    def test_extract_user_id_edge_cases(self):
        """Test user ID extraction from various group ID formats."""
        mock_graphiti = Mock()
        with patch("agno.memory.v2.db.graphiti.Graphiti", return_value=mock_graphiti):
            db = GraphitiMemoryDb(
                uri="bolt://localhost:7687", 
                user="neo4j", 
                password="password",
                table_name="test_table"
            )
            
            # Valid extraction
            user_id = db._extract_user_id("agno_test_table_user_john123")
            assert user_id == "john123"
            
            # Invalid prefix - should return None
            user_id = db._extract_user_id("invalid_prefix_user_john")
            assert user_id is None
            
            # Empty string
            user_id = db._extract_user_id("")
            assert user_id is None

    def test_episode_to_memory_with_invalid_timestamp(self):
        """Test episode conversion with various timestamp formats."""
        mock_graphiti = Mock()
        with patch("agno.memory.v2.db.graphiti.Graphiti", return_value=mock_graphiti):
            db = GraphitiMemoryDb(uri="bolt://localhost:7687", user="neo4j", password="password")
            
            # Invalid timestamp format
            episode = {
                "name": "memory_123",
                "episode_body": json.dumps({"content": "test"}),
                "source_description": "agno_memory_agno_memories",
                "reference_time": "invalid-date-format"
            }
            
            memory = db._episode_to_memory(episode)
            assert memory.last_updated is None

    def test_retry_logic_with_transient_errors(self):
        """Test retry logic for various transient errors."""
        mock_graphiti = Mock()
        with patch("agno.memory.v2.db.graphiti.Graphiti", return_value=mock_graphiti):
            db = GraphitiMemoryDb(uri="bolt://localhost:7687", user="neo4j", password="password")
            
            # Test various transient error patterns
            transient_errors = [
                Exception("Connection timeout"),
                Exception("Network unreachable"),
                Exception("503 Service Unavailable"),
                Exception("Connection reset by peer"),
                Exception("Connection refused")
            ]
            
            for error in transient_errors:
                assert db._is_retriable_error(error) is True
            
            # Non-transient errors
            non_transient_errors = [
                Exception("Authentication failed"),
                Exception("Invalid query"),
                Exception("Permission denied")
            ]
            
            for error in non_transient_errors:
                assert db._is_retriable_error(error) is False