from enum import Enum


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
