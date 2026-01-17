"""Embedding generation using sentence-transformers for semantic search.

This module provides the EmbeddingGenerator class that wraps sentence-transformers
to generate high-quality embeddings for tool descriptions and search queries.

Model: all-MiniLM-L6-v2 (384 dimensions)
- Fast inference
- Good quality for semantic similarity
- Small memory footprint
"""

from __future__ import annotations

import struct
from functools import lru_cache
from typing import TYPE_CHECKING, Any, ClassVar

from loguru import logger
from pydantic import BaseModel, Field, field_validator

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


class EmbeddingConfig(BaseModel):
    """Configuration for embedding generation.
    
    Attributes:
        model_name: Name of the sentence-transformers model to use
        embedding_dim: Dimension of the output embeddings
        max_seq_length: Maximum sequence length for input text
        normalize_embeddings: Whether to L2-normalize output embeddings
        batch_size: Default batch size for batch operations
        show_progress: Whether to show progress bar during batch encoding
    """
    
    model_name: str = Field(
        default="all-MiniLM-L6-v2",
        description="Sentence-transformers model name"
    )
    embedding_dim: int = Field(
        default=384,
        ge=1,
        description="Dimension of output embeddings"
    )
    max_seq_length: int = Field(
        default=256,
        ge=1,
        le=8192,
        description="Maximum sequence length for input text"
    )
    normalize_embeddings: bool = Field(
        default=True,
        description="Whether to L2-normalize embeddings for cosine similarity"
    )
    batch_size: int = Field(
        default=32,
        ge=1,
        le=1024,
        description="Default batch size for encoding"
    )
    show_progress: bool = Field(
        default=False,
        description="Show progress bar during batch encoding"
    )
    
    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, v: str) -> str:
        """Validate that model name is not empty."""
        if not v or not v.strip():
            raise ValueError("Model name cannot be empty")
        return v.strip()


class EmbeddingResult(BaseModel):
    """Result of embedding generation.
    
    Attributes:
        text: Original input text
        embedding: Generated embedding vector
        model: Model used to generate the embedding
        dimension: Dimension of the embedding
    """
    
    text: str = Field(description="Original input text")
    embedding: list[float] = Field(description="Embedding vector")
    model: str = Field(description="Model used for generation")
    dimension: int = Field(description="Embedding dimension")
    
    @property
    def as_bytes(self) -> bytes:
        """Serialize embedding to bytes for BLOB storage."""
        return struct.pack(f"{len(self.embedding)}f", *self.embedding)
    
    @classmethod
    def from_bytes(cls, data: bytes, text: str = "", model: str = "") -> "EmbeddingResult":
        """Deserialize embedding from bytes.
        
        Args:
            data: Bytes containing packed float32 values
            text: Original text (optional, for reconstruction)
            model: Model name (optional, for reconstruction)
            
        Returns:
            EmbeddingResult with deserialized embedding
        """
        num_floats = len(data) // 4  # 4 bytes per float32
        embedding = list(struct.unpack(f"{num_floats}f", data))
        return cls(
            text=text,
            embedding=embedding,
            model=model,
            dimension=num_floats
        )


class EmbeddingGeneratorError(Exception):
    """Base exception for embedding generator errors."""
    pass


class ModelLoadError(EmbeddingGeneratorError):
    """Raised when model loading fails."""
    pass


class EncodingError(EmbeddingGeneratorError):
    """Raised when text encoding fails."""
    pass


class EmbeddingGenerator:
    """Generate embeddings using sentence-transformers.
    
    This class provides methods to generate embeddings for single texts
    and batches of texts using the all-MiniLM-L6-v2 model.
    
    The embeddings are normalized by default for efficient cosine similarity
    calculation using dot product.
    
    Example:
        ```python
        generator = EmbeddingGenerator()
        
        # Single embedding
        embedding = generator.generate_embedding("search for files")
        
        # Batch embeddings
        embeddings = generator.generate_batch([
            "read file contents",
            "write to database",
            "execute shell command"
        ])
        ```
    
    Attributes:
        config: Embedding configuration
        model: Loaded sentence-transformers model (lazy loaded)
    """
    
    # Class-level cache for model instances
    _model_cache: ClassVar[dict[str, Any]] = {}
    
    def __init__(
        self,
        config: EmbeddingConfig | None = None,
        *,
        model_name: str | None = None,
        normalize: bool = True,
    ) -> None:
        """Initialize embedding generator.
        
        Args:
            config: Full configuration object (takes precedence)
            model_name: Model name override (if config not provided)
            normalize: Whether to normalize embeddings (if config not provided)
        """
        if config is not None:
            self.config = config
        else:
            self.config = EmbeddingConfig(
                model_name=model_name or "all-MiniLM-L6-v2",
                normalize_embeddings=normalize,
            )
        
        self._model: SentenceTransformer | None = None
        self._is_loaded = False
        
        logger.debug(
            "EmbeddingGenerator initialized with model={}, dim={}",
            self.config.model_name,
            self.config.embedding_dim
        )
    
    @property
    def model(self) -> "SentenceTransformer":
        """Get or load the sentence-transformers model.
        
        Returns:
            Loaded SentenceTransformer model
            
        Raises:
            ModelLoadError: If model loading fails
        """
        if self._model is None:
            self._load_model()
        return self._model  # type: ignore
    
    @property
    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self._is_loaded
    
    @property
    def embedding_dimension(self) -> int:
        """Get the embedding dimension."""
        return self.config.embedding_dim
    
    def _load_model(self) -> None:
        """Load the sentence-transformers model.
        
        Uses class-level caching to avoid reloading the same model.
        
        Raises:
            ModelLoadError: If model loading fails
        """
        model_name = self.config.model_name
        
        # Check class-level cache first
        if model_name in self._model_cache:
            self._model = self._model_cache[model_name]
            self._is_loaded = True
            logger.debug("Model '{}' loaded from cache", model_name)
            return
        
        try:
            logger.info("Loading sentence-transformers model: {}", model_name)
            
            # Import here to avoid import-time dependency
            from sentence_transformers import SentenceTransformer
            
            self._model = SentenceTransformer(model_name)
            
            # Set max sequence length if configured
            if self.config.max_seq_length:
                self._model.max_seq_length = self.config.max_seq_length
            
            # Cache the model
            self._model_cache[model_name] = self._model
            self._is_loaded = True
            
            logger.info(
                "Model '{}' loaded successfully (dim={})",
                model_name,
                self._model.get_sentence_embedding_dimension()
            )
            
        except ImportError as e:
            logger.error("sentence-transformers not installed: {}", e)
            raise ModelLoadError(
                "sentence-transformers package not installed. "
                "Install with: pip install sentence-transformers"
            ) from e
        except Exception as e:
            logger.error("Failed to load model '{}': {}", model_name, e)
            raise ModelLoadError(f"Failed to load model '{model_name}': {e}") from e
    
    def generate_embedding(self, text: str) -> list[float]:
        """Generate embedding for a single text.
        
        Args:
            text: Input text to encode
            
        Returns:
            List of float values representing the embedding vector
            
        Raises:
            EncodingError: If encoding fails
            ValueError: If text is empty
            
        Example:
            ```python
            embedding = generator.generate_embedding("read file contents")
            print(f"Dimension: {len(embedding)}")  # 384
            ```
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for embedding generation")
            raise ValueError("Text cannot be empty")
        
        try:
            logger.debug("Generating embedding for text: '{}'", text[:50])
            
            embedding = self.model.encode(
                text,
                normalize_embeddings=self.config.normalize_embeddings,
                show_progress_bar=False,
            )
            
            # Convert numpy array to list
            result = embedding.tolist()
            
            logger.debug(
                "Generated embedding with dimension {} for text: '{}'",
                len(result),
                text[:30]
            )
            
            return result
            
        except Exception as e:
            logger.error("Failed to generate embedding: {}", e)
            raise EncodingError(f"Failed to generate embedding: {e}") from e
    
    def generate_embedding_result(self, text: str) -> EmbeddingResult:
        """Generate embedding and return as EmbeddingResult.
        
        Args:
            text: Input text to encode
            
        Returns:
            EmbeddingResult containing the embedding and metadata
            
        Raises:
            EncodingError: If encoding fails
            ValueError: If text is empty
        """
        embedding = self.generate_embedding(text)
        return EmbeddingResult(
            text=text,
            embedding=embedding,
            model=self.config.model_name,
            dimension=len(embedding)
        )
    
    def generate_batch(
        self,
        texts: list[str],
        *,
        batch_size: int | None = None,
        show_progress: bool | None = None,
    ) -> list[list[float]]:
        """Generate embeddings for multiple texts.
        
        Args:
            texts: List of input texts to encode
            batch_size: Override default batch size
            show_progress: Override default progress bar setting
            
        Returns:
            List of embedding vectors (same order as input texts)
            
        Raises:
            EncodingError: If encoding fails
            ValueError: If texts list is empty
            
        Example:
            ```python
            embeddings = generator.generate_batch([
                "read file contents",
                "write to database",
                "execute command"
            ])
            print(f"Generated {len(embeddings)} embeddings")
            ```
        """
        if not texts:
            logger.warning("Empty text list provided for batch embedding")
            raise ValueError("Texts list cannot be empty")
        
        # Filter out empty texts and track indices
        valid_texts: list[tuple[int, str]] = []
        for i, text in enumerate(texts):
            if text and text.strip():
                valid_texts.append((i, text))
        
        if not valid_texts:
            logger.warning("All texts in batch are empty")
            raise ValueError("All texts in batch are empty")
        
        try:
            effective_batch_size = batch_size or self.config.batch_size
            effective_show_progress = (
                show_progress if show_progress is not None 
                else self.config.show_progress
            )
            
            logger.info(
                "Generating embeddings for {} texts (batch_size={})",
                len(valid_texts),
                effective_batch_size
            )
            
            # Extract just the texts for encoding
            texts_to_encode = [t[1] for t in valid_texts]
            
            embeddings = self.model.encode(
                texts_to_encode,
                normalize_embeddings=self.config.normalize_embeddings,
                batch_size=effective_batch_size,
                show_progress_bar=effective_show_progress,
            )
            
            # Convert numpy arrays to lists
            embedding_lists = [emb.tolist() for emb in embeddings]
            
            # Reconstruct full result list with empty embeddings for filtered texts
            result: list[list[float]] = [[] for _ in range(len(texts))]
            for (original_idx, _), embedding in zip(valid_texts, embedding_lists):
                result[original_idx] = embedding
            
            logger.info(
                "Successfully generated {} embeddings",
                len(embedding_lists)
            )
            
            return result
            
        except Exception as e:
            logger.error("Failed to generate batch embeddings: {}", e)
            raise EncodingError(f"Failed to generate batch embeddings: {e}") from e
    
    def generate_batch_results(
        self,
        texts: list[str],
        *,
        batch_size: int | None = None,
        show_progress: bool | None = None,
    ) -> list[EmbeddingResult]:
        """Generate embeddings and return as EmbeddingResult objects.
        
        Args:
            texts: List of input texts to encode
            batch_size: Override default batch size
            show_progress: Override default progress bar setting
            
        Returns:
            List of EmbeddingResult objects (same order as input texts)
            
        Raises:
            EncodingError: If encoding fails
            ValueError: If texts list is empty
        """
        embeddings = self.generate_batch(
            texts,
            batch_size=batch_size,
            show_progress=show_progress
        )
        
        return [
            EmbeddingResult(
                text=text,
                embedding=embedding,
                model=self.config.model_name,
                dimension=len(embedding) if embedding else 0
            )
            for text, embedding in zip(texts, embeddings)
        ]
    
    def preload(self) -> None:
        """Preload the model into memory.
        
        Call this method to load the model before first use,
        avoiding latency on the first embedding generation.
        
        Raises:
            ModelLoadError: If model loading fails
        """
        if not self._is_loaded:
            self._load_model()
            logger.info("Model preloaded successfully")
    
    def unload(self) -> None:
        """Unload the model from memory.
        
        This removes the model reference but keeps it in the class cache.
        To fully unload, use clear_cache().
        """
        self._model = None
        self._is_loaded = False
        logger.debug("Model reference cleared")
    
    @classmethod
    def clear_cache(cls) -> None:
        """Clear the class-level model cache.
        
        This fully removes all cached models from memory.
        """
        cls._model_cache.clear()
        logger.info("Model cache cleared")


@lru_cache(maxsize=1)
def get_default_generator() -> EmbeddingGenerator:
    """Get a singleton default embedding generator.
    
    Returns:
        Cached EmbeddingGenerator instance with default settings
        
    Example:
        ```python
        from mcp_manager.tools.embeddings import get_default_generator
        
        generator = get_default_generator()
        embedding = generator.generate_embedding("search query")
        ```
    """
    return EmbeddingGenerator()


def embedding_to_blob(embedding: list[float]) -> bytes:
    """Convert embedding list to bytes for BLOB storage.
    
    Args:
        embedding: List of float values
        
    Returns:
        Packed bytes (float32)
        
    Example:
        ```python
        blob = embedding_to_blob([0.1, 0.2, 0.3])
        # Store blob in DuckDB BLOB column
        ```
    """
    return struct.pack(f"{len(embedding)}f", *embedding)


def blob_to_embedding(blob: bytes) -> list[float]:
    """Convert BLOB bytes back to embedding list.
    
    Args:
        blob: Packed bytes (float32)
        
    Returns:
        List of float values
        
    Example:
        ```python
        # Retrieve blob from DuckDB
        embedding = blob_to_embedding(blob)
        ```
    """
    num_floats = len(blob) // 4  # 4 bytes per float32
    return list(struct.unpack(f"{num_floats}f", blob))


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Calculate cosine similarity between two embeddings.
    
    For normalized embeddings, this is equivalent to dot product.
    
    Args:
        a: First embedding vector
        b: Second embedding vector
        
    Returns:
        Cosine similarity score (-1 to 1, higher is more similar)
        
    Raises:
        ValueError: If embeddings have different dimensions
        
    Example:
        ```python
        score = cosine_similarity(embedding1, embedding2)
        print(f"Similarity: {score:.4f}")
        ```
    """
    if len(a) != len(b):
        raise ValueError(
            f"Embedding dimensions must match: {len(a)} != {len(b)}"
        )
    
    dot_product = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return dot_product / (norm_a * norm_b)


def dot_product(a: list[float], b: list[float]) -> float:
    """Calculate dot product between two embeddings.
    
    For normalized embeddings (default), this equals cosine similarity.
    
    Args:
        a: First embedding vector
        b: Second embedding vector
        
    Returns:
        Dot product score
        
    Raises:
        ValueError: If embeddings have different dimensions
    """
    if len(a) != len(b):
        raise ValueError(
            f"Embedding dimensions must match: {len(a)} != {len(b)}"
        )
    
    return sum(x * y for x, y in zip(a, b))
