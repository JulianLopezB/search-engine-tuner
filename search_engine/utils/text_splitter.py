from typing import List
import re

class RecursiveCharacterTextSplitter:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, separators: List[str] = ["\n\n", "\n", " ", ""]):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators

    def split_text(self, text: str) -> List[str]:
        chunks = []
        self._split_text_recursive(text, 0, chunks)
        return chunks

    def _split_text_recursive(self, text: str, start: int, chunks: List[str]):
        if len(text) <= self.chunk_size:
            chunks.append(text)
            return

        separator = self._find_separator(text)
        splits = re.split(f"({separator})", text)
        
        current_chunk = ""
        for split in splits:
            if len(current_chunk) + len(split) > self.chunk_size:
                chunks.append(current_chunk)
                current_chunk = split
            else:
                current_chunk += split

            if len(current_chunk) > self.chunk_size - self.chunk_overlap:
                self._split_text_recursive(current_chunk, start, chunks)
                start += len(current_chunk)
                current_chunk = ""

        if current_chunk:
            chunks.append(current_chunk)

    def _find_separator(self, text: str) -> str:
        for separator in self.separators:
            if separator in text:
                return separator
        return self.separators[-1]