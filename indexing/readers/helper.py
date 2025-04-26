from enum import Enum

from llama_index.core import SimpleDirectoryReader
from llama_index.core.readers.base import BaseReader


class Reader(str, Enum):
    """
    Refer to https://llamahub.ai/?tab=readers for a list of readers.

    Attributes:
        SIMPLE_DIRECTORY:
            ('simple_directory')  SimpleDirectoryReader: A reader for reading files from a directory.
    """

    SIMPLE_DIRECTORY = "simple_directory"


PROVIDED_READERS = {Reader.SIMPLE_DIRECTORY: SimpleDirectoryReader}


def prepare_reader(
    reader: Reader | str,
    reader_config: dict,
) -> BaseReader:
    """
    Prepare the transformations for the pipeline.
    """
    reader_class = PROVIDED_READERS.get(reader)
    if reader_class is None:
        raise ValueError(f"Reader {reader} is not supported.")
    reader = reader_class(**reader_config)
    return reader
