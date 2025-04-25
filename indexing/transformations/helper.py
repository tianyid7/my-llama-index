from enum import Enum
from typing import List, Optional

from llama_index.core import Settings
from llama_index.core.extractors import (
    DocumentContextExtractor,
    QuestionsAnsweredExtractor,
    SummaryExtractor,
    TitleExtractor,
)
from llama_index.core.node_parser import (
    CodeSplitter,
    HierarchicalNodeParser,
    HTMLNodeParser,
    JSONNodeParser,
    LangchainNodeParser,
    MarkdownNodeParser,
    SemanticSplitterNodeParser,
    SentenceSplitter,
    SentenceWindowNodeParser,
    SimpleFileNodeParser,
    TokenTextSplitter,
)
from llama_index.core.schema import TransformComponent
from pydantic import BaseModel

from rag.index_manager import IndexManager


class NodeParser(str, Enum):
    """
    Refer to https://docs.llamaindex.ai/en/stable/module_guides/loading/node_parsers/modules/ for a list of node parsers and text splitters.

    Attributes:
        SIMPLE_FILE:
            ('simple_file')  SimpleFileNodeParser: A file-based node parsers. It usually combines with the FlatFileReader automatically use the best node parser for each type of content.
        HTML:
            ('html') HTMLNodeParser: A file-based node parser for HTML content. It will use the BeautifulSoup to parse the HTML content and extract the text.
        JSON:
            ('json') JSONNodeParser: A file-based node parser for JSON content. It will use the json module to parse the JSON content and extract the text.
        MARKDOWN:
            ('markdown') MarkdownNodeParser: A file-based node parser for Markdown content.
        HIERARCHICAL:
            ('hierarchical') HierarchicalNodeParser: A relationship node parser to create parent-child hierarchical nodes. When combined with the AutoMergingRetriever, this enables us to automatically replace retrieved nodes with their parents when a majority of children are retrieved.
        QA:
            ('qa') QANodeParser: A node parser to create question-answer pairs.
        CODE:
            ('code')  CodeSplitter: Splits raw code-text based on the language it is written in.
        LANGCHAIN:
            ('langchain') LangchainNodeParser: Wrap any existing text splitter from langchain with a node parser.
        SENTENCE:
            ('sentence') SentenceSplitter: Split text while respecting the boundaries of sentences.
        SENTENCE_WINDOW:
            ('sentence_window') SentenceWindowNodeParser:  Splits all documents into individual sentences. The resulting nodes also contain the surrounding "window" of sentences around each node in the metadata.
        SEMANTIC_SPLITTER:
            ('semantic_splitter') SemanticSplitterNodeParser: Instead of chunking text with a fixed chunk size, the semantic splitter adaptively picks the breakpoint in-between sentences using embedding similarity. This ensures that a "chunk" contains sentences that are semantically related to each other.
        TOKEN_TEXT:
            ('token_text') TokenTextSplitter: Split to a consistent chunk size according to raw token counts.
    """

    SIMPLE_FILE = "simple_file"
    HTML = "html"
    JSON = "json"
    MARKDOWN = "markdown"
    HIERARCHICAL = "hierarchical"
    QA = "qa"
    CODE = "code"
    LANGCHAIN = "langchain"
    SENTENCE = "sentence"
    SENTENCE_WINDOW = "sentence_window"
    SEMANTIC_SPLITTER = "semantic_splitter"
    TOKEN_TEXT = "token_text"


class MetadataExtractor(str, Enum):
    """
    Refer to https://docs.llamaindex.ai/en/stable/module_guides/loading/documents_and_nodes/usage_metadata_extractor/

    Attributes:
        SUMMARY:
            ('summary')  SummaryExtractor: Automatically extracts a summary over a set of Nodes
        QUESTIONS_ANSWERED:
            ('questions_answered') QuestionsAnsweredExtractor: Extracts a set of questions that each Node can answer
        TITLE:
            ('title') TitleExtractor: Extracts a title over the context of each Node
        DOCUMENT_CONTEXT:
            ('document_context') DocumentContextExtractor: Extracts a 'context' for each node based on the entire document. See https://docs.llamaindex.ai/en/stable/examples/metadata_extraction/DocumentContextExtractor/
    """

    SUMMARY = "summary"
    QUESTIONS_ANSWERED = "questions_answered"
    TITLE = "title"
    DOCUMENT_CONTEXT = "document_context"


PROVIDED_TRANSFORMATIONS = {
    # Node Parsers
    NodeParser.SIMPLE_FILE: SimpleFileNodeParser,
    NodeParser.HTML: HTMLNodeParser,
    NodeParser.JSON: JSONNodeParser,
    NodeParser.MARKDOWN: MarkdownNodeParser,
    NodeParser.HIERARCHICAL: HierarchicalNodeParser,
    # NodeParser.QA: QANodeParser, #TODO: add customized QANodeParser
    NodeParser.CODE: CodeSplitter,
    NodeParser.LANGCHAIN: LangchainNodeParser,
    NodeParser.SENTENCE: SentenceSplitter,
    NodeParser.SENTENCE_WINDOW: SentenceWindowNodeParser,
    NodeParser.SEMANTIC_SPLITTER: SemanticSplitterNodeParser,
    NodeParser.TOKEN_TEXT: TokenTextSplitter,
    # Metadata Extractors
    MetadataExtractor.SUMMARY: SummaryExtractor,
    MetadataExtractor.QUESTIONS_ANSWERED: QuestionsAnsweredExtractor,
    MetadataExtractor.TITLE: TitleExtractor,
    MetadataExtractor.DOCUMENT_CONTEXT: DocumentContextExtractor
    # Custom Transformations
}


class NodeParserParams(BaseModel):
    include_metadata: Optional[bool] = None
    include_prev_next_rel: Optional[bool] = None


class HTMLNodeParserParams(NodeParserParams):
    """
    Configuration for a HTMLNodeParser.
    See https://docs.llamaindex.ai/en/stable/api_reference/node_parsers/html/
    """

    tags: Optional[list[str]] = None


class MarkdownNodeParserParams(NodeParserParams):
    """
    Configuration for a HTMLNodeParser.
    See https://docs.llamaindex.ai/en/stable/api_reference/node_parsers/markdown/
    """

    header_path_separator: Optional[str] = None


class HierarchicalNodeParserParams(NodeParserParams):
    """
    Configuration for a HierarchicalNodeParser.
    See https://docs.llamaindex.ai/en/stable/api_reference/node_parsers/hierarchical/
    """

    chunk_sizes: Optional[list[int]] = [1024, 512, 256]
    node_parser_ids: Optional[list[str]] = None
    node_parser_map: dict[str, str]


class CodeSplitterParams(NodeParserParams):
    """
    Configuration for a CodeSplitter.
    See https://docs.llamaindex.ai/en/stable/api_reference/node_parsers/code/
    """

    language: str
    chunk_lines: Optional[int] = None
    chunk_lines_overlap: Optional[int] = None
    max_chars: Optional[int] = None


class SentenceSplitterParams(NodeParserParams):
    """
    Configuration for a SentenceSplitter.
    See https://docs.llamaindex.ai/en/stable/api_reference/node_parsers/sentence_splitter/
    """

    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None
    separator: Optional[str] = None
    paragraph_separator: Optional[str] = None
    secondary_chunking_regex: Optional[str] = None


class SentenceWindowNodeParserParams(NodeParserParams):
    """
    Configuration for a SentenceWindowNodeParser.
    See https://docs.llamaindex.ai/en/stable/api_reference/node_parsers/sentence_window/
    """

    window_size: Optional[int] = None
    window_metadata_key: Optional[str] = None
    original_text_metadata_key: Optional[str] = None


class SemanticSplitterNodeParserParams(NodeParserParams):
    """
    Configuration for a SemanticSplitterNodeParser.
    See https://docs.llamaindex.ai/en/stable/api_reference/node_parsers/semantic_splitter/

    embed_model: Instead of using OpenAI Embedding model "text-embedding-ada-002" by default.
    See https://github.com/run-llama/llama_index/blob/65a26eb571df2d419903b7ad5945c9634cf5ee34/llama-index-integrations/embeddings/llama-index-embeddings-openai/llama_index/embeddings/openai/base.py#L274C47-L274C65
    Use Settings.embed_model.
    """

    buffer_size: Optional[int] = None
    breakpoint_percentile_threshold: Optional[int] = None
    # embed_model: BaseEmbedding # NOTE: Use Settings.embed_model instead


class TokenTextSplitterParams(NodeParserParams):
    """
    Configuration for a TokenTextSplitter.
    See https://docs.llamaindex.ai/en/stable/api_reference/node_parsers/token_text_splitter/
    """

    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None
    separator: Optional[str] = None
    backup_separators: Optional[list[str]] = None
    keep_whitespaces: Optional[bool] = None


class MetadataExtractorParams(BaseModel):
    """
    Configuration for a MetadataExtractor.
    See https://docs.llamaindex.ai/en/stable/api_reference/extractors/#llama_index.core.extractors.interface.BaseExtractor
    """

    is_text_node_only: Optional[bool] = None
    show_progress: Optional[bool] = None
    metadata_mode: Optional[str] = None
    node_text_template: Optional[str] = None
    disable_template_rewrite: Optional[bool] = None
    in_place: Optional[bool] = None
    num_workers: Optional[int] = None


class SummaryExtractorParams(MetadataExtractorParams):
    """
    Configuration for a SummaryExtractor.
    See https://docs.llamaindex.ai/en/stable/api_reference/extractors/summary/

    llm: Use Settings.llm by default.
    """

    summaries: Optional[List[str]] = [
        "self",
        "prev",
        "next",
    ]  # List of summaries to be extracted: 'self', 'prev', 'next'
    prompt_template: Optional[str] = None


class QuestionsAnsweredExtractorParams(MetadataExtractorParams):
    """
    Configuration for a QuestionsAnsweredExtractor.
    See https://docs.llamaindex.ai/en/stable/api_reference/extractors/question/

    llm: Use Settings.llm by default.
    """

    questions: Optional[List[int]] = None
    prompt_template: Optional[str] = None
    embedding_only: Optional[bool] = None


class TitleExtractorParams(MetadataExtractorParams):
    """
    Configuration for a TitleExtractor.
    See https://docs.llamaindex.ai/en/stable/api_reference/extractors/title/

    llm: Use Settings.llm by default.
    """

    nodes: Optional[int] = None
    node_template: Optional[str] = None
    combine_template: Optional[str] = None
    is_text_node_only: Optional[bool] = None


class DocumentContextExtractorParams(MetadataExtractorParams):
    """
    Configuration for a DocumentContextExtractor.
    See https://docs.llamaindex.ai/en/stable/api_reference/extractors/documentcontext/

    llm: Use Settings.llm by default.
    docstore: NOTE!!! This is required but will be provided by the transformation helper function.
    """

    max_context_length: int
    # docstore: KVDocumentStore  # NOTE: This is required but will be provided by the transformation helper function.
    oversized_document_strategy: Optional[str] = None
    max_output_tokens: Optional[int] = None
    key: Optional[str] = None
    prompt: Optional[str] = None


class TransformationConfig(BaseModel):
    """
    Configuration for transformations.
    """

    html: Optional[HTMLNodeParserParams] = None
    markdown: Optional[MarkdownNodeParserParams] = None
    hierarchical: Optional[HierarchicalNodeParserParams] = None
    code: Optional[CodeSplitterParams] = None
    sentence: Optional[SentenceSplitterParams] = None
    sentence_window: Optional[SentenceWindowNodeParserParams] = None
    semantic_splitter: Optional[SemanticSplitterNodeParserParams] = None
    token_text: Optional[TokenTextSplitterParams] = None
    summary: Optional[SummaryExtractorParams] = None
    questions_answered: Optional[QuestionsAnsweredExtractorParams] = None
    title: Optional[TitleExtractorParams] = None
    document_context: Optional[DocumentContextExtractorParams] = None


def prepare_transformations(
    transformations: List[NodeParser | MetadataExtractor | str],
    config: TransformationConfig,
) -> List[TransformComponent]:
    """
    Prepare the transformations.

    Args:
        transformations: List of transformation names. Order matters.
        config: Configuration for the transformations.
    """
    result = []
    for t in transformations:
        t = t.lower()
        transformation_class = getattr(PROVIDED_TRANSFORMATIONS, t)
        transformation_config = getattr(config, t, None)
        if transformation_config:
            if t == NodeParser.SEMANTIC_SPLITTER:
                # Use Settings.embed_model instead of the default OpenAI embedding model.
                transformation = transformation_class(
                    **{
                        **transformation_config.dict(),
                        **{"embed_model": Settings.embed_model},
                    }
                )
            elif t == MetadataExtractor.DOCUMENT_CONTEXT:
                # Provide docstore
                transformation = transformation_class(
                    **{
                        **transformation_config.dict(),
                        **{"docstore": IndexManager().base_index.docstore},
                    }
                )
            else:
                transformation = transformation_class(**transformation_config.dict())
        else:
            transformation = transformation_class()
        result.append(transformation)

    return result
