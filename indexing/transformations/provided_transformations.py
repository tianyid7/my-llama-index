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

from indexing.transformations.metadata_extractors.metadata_extractor import (
    MetadataExtractor,
)
from indexing.transformations.node_parsers.node_parser import NodeParser

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
