from llama_index.core import SimpleDirectoryReader

from indexing.readers.reader import Reader

PROVIDED_READERS = {Reader.SIMPLE_DIRECTORY: SimpleDirectoryReader}
