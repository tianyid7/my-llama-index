from enum import Enum


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
