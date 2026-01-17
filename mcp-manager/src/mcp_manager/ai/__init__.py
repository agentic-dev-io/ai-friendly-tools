"""AI features for MCP Manager: embeddings, NLP, examples, LLM export, workflows."""

from .embeddings import (
    EmbeddingModel,
    EmbeddingCache,
    ToolEmbedder,
    SemanticSearcher,
)
from .nlp import (
    IntentClassifier,
    QueryParser,
    NaturalLanguageProcessor,
    Intent,
    ParsedQuery,
)
from .examples import (
    ExampleGenerator,
    UsageExample,
    ExampleTemplate,
    SmartExampleBuilder,
)
from .llm_export import (
    LLMExporter,
    ExportFormat,
    ToolDocumentation,
    ExportConfig,
)
from .suggestions import (
    ToolSuggester,
    SuggestionContext,
    Suggestion,
    WorkflowSuggester,
)
from .workflows import (
    Workflow,
    WorkflowStep,
    WorkflowRunner,
    WorkflowBuilder,
    WorkflowRegistry,
    WorkflowRecommender,
    StepStatus,
    WorkflowStatus,
    StepResult,
    WorkflowResult,
)

__all__ = [
    # Embeddings
    "EmbeddingModel",
    "EmbeddingCache",
    "ToolEmbedder",
    "SemanticSearcher",
    # NLP
    "IntentClassifier",
    "QueryParser",
    "NaturalLanguageProcessor",
    "Intent",
    "ParsedQuery",
    # Examples
    "ExampleGenerator",
    "UsageExample",
    "ExampleTemplate",
    "SmartExampleBuilder",
    # LLM Export
    "LLMExporter",
    "ExportFormat",
    "ToolDocumentation",
    "ExportConfig",
    # Suggestions
    "ToolSuggester",
    "SuggestionContext",
    "Suggestion",
    "WorkflowSuggester",
    # Workflows
    "Workflow",
    "WorkflowStep",
    "WorkflowRunner",
    "WorkflowBuilder",
    "WorkflowRegistry",
    "WorkflowRecommender",
    "StepStatus",
    "WorkflowStatus",
    "StepResult",
    "WorkflowResult",
]
