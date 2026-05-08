"""Simple summarization utilities."""

from __future__ import annotations

from coding_agent.memory.retrieval import RetrievalBundle


def summarize_retrieval(bundle: RetrievalBundle) -> str:
    """Create compact retrieval text for prompt building."""
    parts: list[str] = []
    if bundle.memory_hits:
        parts.append("Relevant memories:\n" + "\n".join(f"- {item}" for item in bundle.memory_hits))
    if bundle.file_chunks:
        chunk_text = "\n\n".join(
            f"[{chunk.path}:{chunk.start_line}-{chunk.end_line}]\n{chunk.content}" for chunk in bundle.file_chunks
        )
        parts.append("Relevant code:\n" + chunk_text)
    return "\n\n".join(parts)

