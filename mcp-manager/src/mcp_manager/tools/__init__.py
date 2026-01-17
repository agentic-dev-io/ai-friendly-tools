"""Tools management module for MCP manager."""

from .schema import (
    ToolSchema,
    ServerSchema,
    ToolHistorySchema,
    SchemaManager,
    ToolRegistry,
)
from .validation import (
    ToolValidator,
    ToolErrorSuggester,
    ValidationResult,
    ValidationError_,
)
from .execution import (
    ToolExecutor,
    ExecutionContext,
    ExecutionResult,
    BatchExecutor,
)
from .search import ToolSearcher
from .embeddings import (
    EmbeddingGenerator,
    EmbeddingConfig,
    EmbeddingResult,
    EmbeddingGeneratorError,
    ModelLoadError,
    EncodingError,
    get_default_generator,
    embedding_to_blob,
    blob_to_embedding,
    cosine_similarity,
    dot_product,
)
from .vector_store import (
    VectorStore,
    VectorStoreConfig,
    VectorSearchResult,
    EmbeddingRecord,
    VectorStoreError,
    SimilarityMetric,
    create_vector_store,
)

__all__ = [
    # Schema
    "ToolSchema",
    "ServerSchema",
    "ToolHistorySchema",
    "SchemaManager",
    "ToolRegistry",
    # Validation
    "ToolValidator",
    "ToolErrorSuggester",
    "ValidationResult",
    "ValidationError_",
    # Execution
    "ToolExecutor",
    "ExecutionContext",
    "ExecutionResult",
    "BatchExecutor",
    # Search
    "ToolSearcher",
    # Embeddings
    "EmbeddingGenerator",
    "EmbeddingConfig",
    "EmbeddingResult",
    "EmbeddingGeneratorError",
    "ModelLoadError",
    "EncodingError",
    "get_default_generator",
    "embedding_to_blob",
    "blob_to_embedding",
    "cosine_similarity",
    "dot_product",
    # Vector Store
    "VectorStore",
    "VectorStoreConfig",
    "VectorSearchResult",
    "EmbeddingRecord",
    "VectorStoreError",
    "SimilarityMetric",
    "create_vector_store",
]
