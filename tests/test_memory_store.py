from pathlib import Path

from coding_agent.memory.models import MemoryRecord, RunRecord
from coding_agent.memory.store import MemoryStore


def test_memory_store_persists_records(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memory.db")
    store.add_memory(MemoryRecord(kind="semantic", content="auth module uses JWT", tags=["auth"]))
    store.record_run(RunRecord(run_id="run-1", goal="test", status="completed"))
    assert store.search_memory("JWT")
    assert store.list_runs()

