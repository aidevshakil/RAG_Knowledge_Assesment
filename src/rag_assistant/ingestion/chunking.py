"""Recursive character chunker with overlap.

Splits on the most semantic separator that fits (paragraph → line → sentence →
word → character), so chunks rarely cut mid-thought. Overlap carries context
across boundaries, which measurably improves recall for questions whose answer
straddles two chunks.

Implemented here rather than pulled from LangChain: it is ~80 lines, keeps the
dependency surface small, and lets us track character offsets so each chunk can
report its source page.
"""

from __future__ import annotations

from dataclasses import dataclass

from rag_assistant.config import ChunkingConfig
from rag_assistant.core.types import Chunk, new_id
from rag_assistant.ingestion.loaders import LoadedDocument

SEPARATORS: tuple[str, ...] = ("\n\n", "\n", ". ", "; ", ", ", " ", "")


@dataclass(slots=True)
class _Piece:
    text: str
    start: int


class RecursiveChunker:
    def __init__(self, config: ChunkingConfig) -> None:
        self.config = config

    def split(
        self, document: LoadedDocument, *, document_id: str, document_name: str
    ) -> list[Chunk]:
        """Chunk a loaded document, attaching page numbers where known."""
        chunks: list[Chunk] = []
        for index, piece in enumerate(self._merge(self._split(document.text, 0))):
            metadata: dict = {"char_start": piece.start, "chars": len(piece.text)}
            page = document.page_for_offset(piece.start)
            if page is not None:
                metadata["page"] = page
            chunks.append(
                Chunk(
                    id=new_id(),
                    document_id=document_id,
                    document_name=document_name,
                    text=piece.text,
                    index=index,
                    metadata=metadata,
                )
            )
        return chunks

    def split_text(
        self, text: str, *, document_id: str = "adhoc", name: str = "text"
    ) -> list[Chunk]:
        return self.split(
            LoadedDocument(text=text, source_type="raw"),
            document_id=document_id,
            document_name=name,
        )

    def _split(self, text: str, offset: int, depth: int = 0) -> list[_Piece]:
        """Break `text` into pieces no longer than chunk_size, recursively."""
        text = text.strip()
        if not text:
            return []
        if len(text) <= self.config.chunk_size:
            return [_Piece(text, offset)]

        separator = SEPARATORS[min(depth, len(SEPARATORS) - 1)]
        if separator == "":
            size = self.config.chunk_size
            return [_Piece(text[i : i + size], offset + i) for i in range(0, len(text), size)]

        pieces: list[_Piece] = []
        cursor = 0
        for part in text.split(separator):
            start = offset + cursor
            cursor += len(part) + len(separator)
            if not part.strip():
                continue
            if len(part) > self.config.chunk_size:
                pieces.extend(self._split(part, start, depth + 1))
            else:
                pieces.append(_Piece(part, start))
        return pieces

    def _merge(self, pieces: list[_Piece]) -> list[_Piece]:
        """Greedily pack pieces up to chunk_size, then re-seed with overlap."""
        if not pieces:
            return []
        size, overlap = self.config.chunk_size, self.config.chunk_overlap
        merged: list[_Piece] = []
        buffer: list[_Piece] = []
        length = 0

        def flush() -> None:
            nonlocal buffer, length
            if not buffer:
                return
            merged.append(_Piece(" ".join(p.text for p in buffer).strip(), buffer[0].start))
            carry: list[_Piece] = []
            carried = 0
            for piece in reversed(buffer):
                if carried + len(piece.text) > overlap:
                    break
                carry.insert(0, piece)
                carried += len(piece.text) + 1
            buffer = carry
            length = carried

        for piece in pieces:
            if length + len(piece.text) + 1 > size and buffer:
                flush()
            buffer.append(piece)
            length += len(piece.text) + 1
        flush()

        coalesced: list[_Piece] = []
        for piece in merged:
            if len(piece.text) < self.config.min_chunk_chars and coalesced:
                previous = coalesced[-1]
                coalesced[-1] = _Piece(f"{previous.text} {piece.text}".strip(), previous.start)
            else:
                coalesced.append(piece)
        return coalesced
