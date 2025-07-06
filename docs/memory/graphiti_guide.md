# GraphitiMemoryDb User Guide

GraphitiMemoryDb is a graph database implementation of Agno's MemoryDb interface, powered by [Graphiti](https://github.com/getzep/graphiti) and Neo4j. It provides graph-based memory storage with vector similarity search and relationship modeling capabilities.

## Features

- **Graph-based storage**: Leverages Neo4j's graph database for complex relationship modeling
- **Vector similarity search**: Built-in support for semantic memory retrieval
- **Multi-user isolation**: Secure memory separation between users
- **Batch operations**: Efficient bulk insert and delete operations
- **Retry logic**: Automatic retry with exponential backoff for transient failures
- **Configurable limits**: Adjustable result limits and JSON payload sizes

## Installation

Install Agno with Graphiti support:

```bash
pip install agno[graphiti]
```

### Prerequisites

1. **Neo4j Database**: GraphitiMemoryDb requires a Neo4j instance (version 5.x recommended)
   
   Using Docker:
   ```bash
   docker run -d \
     --name neo4j \
     -p 7474:7474 -p 7687:7687 \
     -e NEO4J_AUTH=neo4j/password \
     -e NEO4J_PLUGINS='["apoc", "graph-data-science"]' \
     neo4j:5.15
   ```

2. **Environment Variables** (optional):
   ```bash
   export NEO4J_URI="bolt://localhost:7687"
   export NEO4J_USER="neo4j"
   export NEO4J_PASSWORD="password"
   export OPENAI_API_KEY="your-api-key"  # For embeddings
   ```

## Quick Start

```python
from agno.memory.v2.db.graphiti import GraphitiMemoryDb
from agno.memory.v2.db.schema import MemoryRow
from datetime import datetime

# Initialize the memory database
memory_db = GraphitiMemoryDb(
    uri="bolt://localhost:7687",
    user="neo4j",
    password="password",
    table_name="agent_memories"
)

# Create the database
memory_db.create()

# Create a memory
memory = MemoryRow(
    memory={
        "content": "User prefers dark mode in applications",
        "context": "UI preferences",
        "confidence": 0.95
    },
    user_id="user_123",
    last_updated=datetime.now()
)

# Store the memory
result = memory_db.upsert_memory(memory)
print(f"Stored memory with ID: {result.id}")

# Retrieve memories for a user
user_memories = memory_db.read_memories(user_id="user_123")
for mem in user_memories:
    print(f"Memory: {mem.memory}")

# Semantic search
similar_memories = memory_db.similarity_search(
    query="What are the user's interface preferences?",
    user_id="user_123",
    limit=5
)
```

## Configuration Options

### Constructor Parameters

```python
GraphitiMemoryDb(
    uri: str,                    # Neo4j connection URI
    user: str,                   # Database username
    password: str,               # Database password
    table_name: str = "agno_memories",  # Logical table name
    llm_client: Optional[LLMClient] = None,  # Custom LLM client
    embedder: Optional[EmbedderClient] = None,  # Custom embedder
    max_results: int = 1000,     # Max results per query
    max_json_size_mb: int = 1,   # Max JSON payload size in MB
    **kwargs                     # Additional Graphiti options
)
```

### Advanced Configuration

```python
# Custom LLM and embedder clients
from graphiti_core.llm_client import OpenAIClient
from graphiti_core.embedder.client import OpenAIEmbedderClient

llm_client = OpenAIClient(model="gpt-4")
embedder = OpenAIEmbedderClient(model="text-embedding-3-small")

memory_db = GraphitiMemoryDb(
    uri="bolt://localhost:7687",
    user="neo4j",
    password="password",
    table_name="advanced_memories",
    llm_client=llm_client,
    embedder=embedder,
    max_results=5000,
    max_json_size_mb=10
)
```

## API Reference

### Core Methods

#### `create() -> None`
Initialize the database and create necessary indices.

#### `upsert_memory(memory: MemoryRow) -> Optional[MemoryRow]`
Insert or update a memory. Auto-generates ID and timestamp if not provided.

#### `read_memories(user_id: Optional[str] = None, limit: Optional[int] = None, sort: Optional[str] = None) -> List[MemoryRow]`
Retrieve memories with optional filtering and sorting.

- `user_id`: Filter by user
- `limit`: Maximum number of results
- `sort`: "asc" or "desc" (default: "desc" by timestamp)

#### `similarity_search(query: str, user_id: Optional[str] = None, limit: Optional[int] = None) -> List[MemoryRow]`
Perform semantic similarity search on memories.

#### `delete_memory(memory_id: str) -> None`
Delete a specific memory by ID.

#### `memory_exists(memory: MemoryRow) -> bool`
Check if a memory exists.

#### `clear() -> bool`
Remove all memories from the table.

#### `drop_table() -> None`
Delete the entire table and all its memories.

### Batch Operations

#### `bulk_upsert_memories(memories: List[MemoryRow]) -> List[MemoryRow]`
Efficiently insert or update multiple memories.

```python
memories = [
    MemoryRow(memory={"content": f"Memory {i}"}, user_id="user_123")
    for i in range(100)
]
successful = memory_db.bulk_upsert_memories(memories)
print(f"Successfully stored {len(successful)} memories")
```

#### `bulk_delete_memories(memory_ids: List[str]) -> int`
Delete multiple memories by their IDs.

```python
ids_to_delete = ["mem_1", "mem_2", "mem_3"]
deleted_count = memory_db.bulk_delete_memories(ids_to_delete)
print(f"Deleted {deleted_count} memories")
```

## Best Practices

### 1. Connection Management
GraphitiMemoryDb handles connection pooling internally. For long-running applications:

```python
# Initialize once at application startup
memory_db = GraphitiMemoryDb(...)
memory_db.create()

# Use throughout application lifecycle
# No need to close connections - handled automatically
```

### 2. Error Handling
The implementation includes automatic retry logic for transient failures:

```python
try:
    memory_db.upsert_memory(memory)
except RuntimeError as e:
    # Permanent failure after retries
    print(f"Failed to store memory: {e}")
```

### 3. Memory Structure
Design memory objects for optimal graph relationships:

```python
memory = MemoryRow(
    memory={
        "content": "Main information",
        "metadata": {
            "category": "preference",
            "source": "conversation",
            "confidence": 0.9
        },
        "relationships": ["related_memory_id_1", "related_memory_id_2"]
    },
    user_id="user_123"
)
```

### 4. Performance Optimization

For large-scale operations:

```python
# Use batch operations for multiple memories
memories = [...]  # List of 1000 memories
memory_db.bulk_upsert_memories(memories)

# Limit search results appropriately
results = memory_db.similarity_search(
    query="user preferences",
    limit=10  # Only get top 10 most relevant
)

# Configure appropriate limits
memory_db = GraphitiMemoryDb(
    ...,
    max_results=500,  # Reduce for better performance
    max_json_size_mb=5  # Increase for larger payloads
)
```

## Troubleshooting

### Connection Issues

```python
# Test connection
try:
    memory_db.create()
    print("Connected successfully")
except RuntimeError as e:
    print(f"Connection failed: {e}")
    # Check Neo4j is running and credentials are correct
```

### Performance Issues

1. **Slow queries**: Add indices in Neo4j for frequently queried fields
2. **Memory usage**: Reduce `max_results` or implement pagination
3. **Network latency**: Consider co-locating application with Neo4j

### Common Errors

- `ImportError`: Install with `pip install agno[graphiti]`
- `Connection refused`: Ensure Neo4j is running and accessible
- `Authentication failed`: Verify Neo4j credentials
- `JSON too large`: Increase `max_json_size_mb` or reduce payload size

## Integration with Agno Agents

```python
from agno import Agent
from agno.memory.agent import AgentMemory
from agno.memory.v2.db.graphiti import GraphitiMemoryDb

# Create memory database
memory_db = GraphitiMemoryDb(
    uri="bolt://localhost:7687",
    user="neo4j",
    password="password"
)

# Create agent with memory
agent = Agent(
    name="assistant",
    instructions="You are a helpful assistant with memory capabilities.",
    memory=AgentMemory(db=memory_db)
)

# Agent will now use GraphitiMemoryDb for memory storage
```

## Migration from Other Providers

Migrating from SQLite or other providers:

```python
from agno.memory.v2.db.sqlite import SqliteMemoryDb
from agno.memory.v2.db.graphiti import GraphitiMemoryDb

# Source database
source_db = SqliteMemoryDb(db_file="memories.db")

# Target database
target_db = GraphitiMemoryDb(
    uri="bolt://localhost:7687",
    user="neo4j",
    password="password"
)
target_db.create()

# Migrate memories
memories = source_db.read_memories()
successful = target_db.bulk_upsert_memories(memories)
print(f"Migrated {len(successful)} memories")
```

## Advanced Features

### Custom Embeddings

```python
from graphiti_core.embedder.client import EmbedderClient

class CustomEmbedder(EmbedderClient):
    async def create(self, text: str) -> List[float]:
        # Your custom embedding logic
        return embeddings

memory_db = GraphitiMemoryDb(
    ...,
    embedder=CustomEmbedder()
)
```

### Graph Traversal

Access the underlying Graphiti client for advanced graph operations:

```python
# Direct Graphiti access
graphiti_client = memory_db.graphiti

# Custom graph queries (use with caution)
# Refer to Graphiti documentation for advanced usage
```

## Performance Benchmarks

Based on our testing with 50 memories:

| Operation | Time | Notes |
|-----------|------|-------|
| Create DB | ~0.5s | One-time initialization |
| Bulk Insert (50) | ~2.5s | Includes embedding generation |
| Read All | ~0.3s | No filtering |
| Similarity Search | ~0.4s | Vector similarity |
| Clear | ~1.2s | Delete all memories |

*Note: Performance varies based on Neo4j configuration and hardware.*

## Conclusion

GraphitiMemoryDb provides a powerful graph-based memory storage solution for Agno agents. Its combination of graph relationships and vector similarity search makes it ideal for applications requiring complex memory structures and semantic retrieval capabilities.