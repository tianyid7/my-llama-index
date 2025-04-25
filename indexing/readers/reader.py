from enum import Enum


class Reader(str, Enum):
    """
    Refer to https://llamahub.ai/?tab=readers for a list of readers.

    Attributes:
        SIMPLE_DIRECTORY:
            ('simple_directory')  SimpleDirectoryReader: A reader for reading files from a directory.
    """

    SIMPLE_DIRECTORY = "simple_directory"
