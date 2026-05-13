"""Deterministic repository scan for project understanding."""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path
import json
import tempfile

from pydantic import BaseModel, Field

from coding_agent.sandbox.workspace_guard import WorkspaceGuard


class ModuleEdge(BaseModel):
    source: str
    target: str


class RepoScanResult(BaseModel):
    languages: list[str] = Field(default_factory=list)
    top_directories: list[str] = Field(default_factory=list)
    important_files: list[str] = Field(default_factory=list)
    confirmed_frameworks: list[str] = Field(default_factory=list)
    possible_frameworks: list[str] = Field(default_factory=list)
    entrypoints: list[str] = Field(default_factory=list)
    critical_modules: list[str] = Field(default_factory=list)
    api_surfaces: list[str] = Field(default_factory=list)
    storage_backends: list[str] = Field(default_factory=list)
    llm_dependencies: list[str] = Field(default_factory=list)
    security_boundaries: list[str] = Field(default_factory=list)
    module_edges: list[ModuleEdge] = Field(default_factory=list)

    def to_prompt_text(self) -> str:
        sections = [
            "Languages: " + (", ".join(self.languages) or "none"),
            "Top directories: " + (", ".join(self.top_directories) or "none"),
            "Important files: " + (", ".join(self.important_files) or "none"),
            "Confirmed frameworks: " + (", ".join(self.confirmed_frameworks) or "none"),
            "Possible frameworks: " + (", ".join(self.possible_frameworks) or "none"),
            "Entrypoints: " + (", ".join(self.entrypoints) or "none"),
            "Critical modules: " + (", ".join(self.critical_modules) or "none"),
            "API surfaces: " + (", ".join(self.api_surfaces) or "none"),
            "Storage backends: " + (", ".join(self.storage_backends) or "none"),
            "LLM dependencies: " + (", ".join(self.llm_dependencies) or "none"),
            "Security boundaries: " + (", ".join(self.security_boundaries) or "none"),
            "Module graph:\n" + ("\n".join(f"{edge.source} -> {edge.target}" for edge in self.module_edges) or "none"),
        ]
        return "\n\n".join(sections)


class RepoScanner:
    """Build a deterministic repository map from the workspace."""

    def __init__(self, guard: WorkspaceGuard):
        self.guard = guard

    def scan(self) -> RepoScanResult:
        files = self.guard.glob(".", "*")
        rel_paths = [path.relative_to(self.guard.root).as_posix() for path in files]
        source_files = self._prioritized_source_files(files)
        critical_modules = self._collect_critical_modules(rel_paths)
        analysis_files = self._select_analysis_files(source_files, critical_modules)
        entrypoints = self._collect_entrypoints(analysis_files)
        confirmed_frameworks = self._collect_confirmed_frameworks(analysis_files, entrypoints, critical_modules)
        return RepoScanResult(
            languages=self._collect_languages(rel_paths),
            top_directories=self._collect_top_directories(rel_paths),
            important_files=self._collect_important_files(rel_paths),
            confirmed_frameworks=confirmed_frameworks,
            possible_frameworks=self._collect_possible_frameworks(source_files, confirmed_frameworks),
            entrypoints=entrypoints,
            critical_modules=critical_modules,
            api_surfaces=self._collect_api_surfaces(analysis_files),
            storage_backends=self._collect_storage_backends(analysis_files),
            llm_dependencies=self._collect_llm_dependencies(analysis_files),
            security_boundaries=self._collect_security_boundaries(rel_paths),
            module_edges=self._collect_module_edges(source_files),
        )

    def default_artifact_path(self) -> Path:
        return self.guard.root / "artifacts" / "repo_scan.json"

    def save_artifact(self, target: Path | None = None) -> Path:
        artifact_path = target or self.default_artifact_path()
        payload = self.scan().model_dump(mode="json")
        serialized = json.dumps(payload, indent=2, ensure_ascii=False)
        try:
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            artifact_path.write_text(serialized, encoding="utf-8")
            return artifact_path
        except PermissionError:
            fallback = self._fallback_artifact_path()
            fallback.parent.mkdir(parents=True, exist_ok=True)
            fallback.write_text(serialized, encoding="utf-8")
            return fallback

    def load_artifact(self, target: Path | None = None) -> RepoScanResult | None:
        loaded = self.load_artifact_with_path(target)
        return loaded[1] if loaded else None

    def load_artifact_with_path(self, target: Path | None = None) -> tuple[Path, RepoScanResult] | None:
        candidates = [target] if target else [self.default_artifact_path(), self._fallback_artifact_path()]
        for artifact_path in candidates:
            if artifact_path is None or not artifact_path.exists():
                continue
            try:
                payload = json.loads(artifact_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            return artifact_path, RepoScanResult.model_validate(payload)
        return None

    def _fallback_artifact_path(self) -> Path:
        return Path(tempfile.gettempdir()) / "coding-agent" / "repo-scan" / self.guard.root.name / "repo_scan.json"

    @staticmethod
    def _collect_languages(rel_paths: list[str]) -> list[str]:
        extensions = Counter(path.rsplit(".", 1)[-1].lower() for path in rel_paths if "." in path.rsplit("/", 1)[-1])
        return [f"{ext}={count}" for ext, count in extensions.most_common(8)]

    @staticmethod
    def _collect_top_directories(rel_paths: list[str]) -> list[str]:
        top_dirs = Counter(path.split("/", 1)[0] for path in rel_paths if "/" in path)
        return [f"{name}={count}" for name, count in top_dirs.most_common(10)]

    @staticmethod
    def _collect_important_files(rel_paths: list[str]) -> list[str]:
        important = []
        exact = {"README.md", "pyproject.toml", "package.json", "requirements.txt", "Cargo.toml", "go.mod", "Dockerfile"}
        suffixes = (
            "/main.py",
            "/app.py",
            "/server.py",
            "/index.ts",
            "/index.js",
            "/cli/main.py",
            "/api/app.py",
            "/core/orchestrator.py",
            "/sandbox/workspace_guard.py",
            "/sandbox/command_policy.py",
        )
        for path in rel_paths:
            if path in exact or path.endswith(suffixes):
                important.append(path)
        return important[:20]

    @staticmethod
    def _prioritized_source_files(files: list[Path]) -> list[Path]:
        def score(path: Path) -> tuple[int, str]:
            rel = path.as_posix().lower()
            if "/tests/" in rel or rel.startswith("tests/"):
                return (3, rel)
            if "/src/" in rel or rel.startswith("src/"):
                return (0, rel)
            if rel.startswith("config/"):
                return (1, rel)
            return (2, rel)

        return sorted(
            [path for path in files if path.suffix.lower() in {".py", ".ts", ".js", ".tsx", ".jsx"}],
            key=score,
        )

    def _collect_confirmed_frameworks(self, files: list[Path], entrypoints: list[str], critical_modules: list[str]) -> list[str]:
        framework_targets = {
            "FastAPI": {"fastapi"},
            "Typer": {"typer"},
            "Flask": {"flask"},
            "Django": {"django"},
            "Pytest": {"pytest"},
            "Pydantic": {"pydantic", "pydantic_settings"},
            "SQLAlchemy": {"sqlalchemy"},
            "React": {"react"},
            "Vite": {"vite"},
            "Express": {"express"},
            "Ollama": {"ollama", "coding_agent.llm.ollama_provider"},
        }
        found: list[str] = []
        allowed_paths = set(entrypoints) | set(critical_modules)
        for path in files[:20]:
            rel = path.as_posix().lower()
            if rel.endswith("/repo_scan.py") or rel.endswith("\\repo_scan.py"):
                continue
            workspace_rel = path.relative_to(self.guard.root).as_posix()
            if workspace_rel not in allowed_paths:
                continue
            try:
                content = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            imports = set(self._extract_import_targets(content))
            lowered_content = content.lower()
            for label, targets in framework_targets.items():
                if label in found:
                    continue
                if any(target in imports for target in targets):
                    found.append(label)
                    continue
                if label == "Ollama" and ("ollama" in lowered_content or "ollama_provider" in lowered_content):
                    found.append(label)
        return found

    @staticmethod
    def _collect_possible_frameworks(files: list[Path], confirmed_frameworks: list[str]) -> list[str]:
        patterns = {
            "Flask": re.compile(r"\bflask\b", flags=re.IGNORECASE),
            "Django": re.compile(r"\bdjango\b", flags=re.IGNORECASE),
            "React": re.compile(r"\breact\b", flags=re.IGNORECASE),
            "Vite": re.compile(r"\bvite\b", flags=re.IGNORECASE),
            "Express": re.compile(r"\bexpress\b", flags=re.IGNORECASE),
            "SQLAlchemy": re.compile(r"\bsqlalchemy\b", flags=re.IGNORECASE),
        }
        found: list[str] = []
        for path in files[:24]:
            rel = path.as_posix().lower()
            if rel.endswith("/repo_scan.py") or rel.endswith("\\repo_scan.py"):
                continue
            try:
                content = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            for label, pattern in patterns.items():
                if label in found or label in confirmed_frameworks:
                    continue
                if pattern.search(content):
                    found.append(label)
        return found

    def _select_analysis_files(self, source_files: list[Path], critical_modules: list[str]) -> list[Path]:
        selected: list[Path] = []
        critical_set = set(critical_modules)
        for path in source_files:
            rel = path.relative_to(self.guard.root).as_posix()
            if rel.endswith("/core/repo_scan.py"):
                continue
            if rel in critical_set or rel.endswith(("/api/app.py", "/cli/main.py", "/config/settings.py", "/core/orchestrator.py")):
                selected.append(path)
        for path in source_files:
            if path in selected:
                continue
            rel = path.relative_to(self.guard.root).as_posix()
            if rel.endswith("/core/repo_scan.py"):
                continue
            if rel.startswith("src/") and len(selected) < 16:
                selected.append(path)
        return selected[:16]

    @staticmethod
    def _collect_critical_modules(rel_paths: list[str]) -> list[str]:
        keywords = (
            "core/orchestrator.py",
            "api/app.py",
            "cli/main.py",
            "sandbox/workspace_guard.py",
            "sandbox/command_policy.py",
            "memory/store.py",
            "llm/ollama_provider.py",
            "llm/prompt_builder.py",
            "tools/file_tools.py",
            "tools/git_tools.py",
        )
        critical = [path for path in rel_paths if any(token in path for token in keywords)]
        return critical[:12]

    def _collect_api_surfaces(self, files: list[Path]) -> list[str]:
        surfaces: list[str] = []
        route_pattern = re.compile(r'@app\.(get|post|put|delete|patch)\(\s*["\']([^"\']+)["\']', flags=re.IGNORECASE)
        for path in files:
            rel = path.relative_to(self.guard.root).as_posix()
            if not rel.endswith(".py"):
                continue
            try:
                content = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            for method, route in route_pattern.findall(content):
                surfaces.append(f"{method.upper()} {route} ({rel})")
        return surfaces[:12]

    def _collect_storage_backends(self, files: list[Path]) -> list[str]:
        found: list[str] = []
        checks = {
            "SQLite": re.compile(r"\bsqlite3\b|\.db\b", flags=re.IGNORECASE),
            "PostgreSQL": re.compile(r"\bpostgres\b|\bpsycopg\b", flags=re.IGNORECASE),
            "MySQL": re.compile(r"\bmysql\b", flags=re.IGNORECASE),
            "Redis": re.compile(r"\bredis\b", flags=re.IGNORECASE),
        }
        for path in files:
            try:
                content = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            for label, pattern in checks.items():
                if label in found:
                    continue
                if pattern.search(content):
                    found.append(label)
        return found

    def _collect_llm_dependencies(self, files: list[Path]) -> list[str]:
        found: list[str] = []
        checks = {
            "OllamaProvider": re.compile(r"\bOllamaProvider\b"),
            "OpenAICompatibleProvider": re.compile(r"\bOpenAICompatibleProvider\b"),
            "httpx": re.compile(r"\bhttpx\b"),
            "ollama_base_url": re.compile(r"127\.0\.0\.1:11434|ollama", flags=re.IGNORECASE),
        }
        for path in files:
            try:
                content = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            for label, pattern in checks.items():
                if label in found:
                    continue
                if pattern.search(content):
                    found.append(label)
        return found

    @staticmethod
    def _collect_security_boundaries(rel_paths: list[str]) -> list[str]:
        boundaries: list[str] = []
        checks = (
            ("WorkspaceGuard", "sandbox/workspace_guard.py"),
            ("CommandPolicyEngine", "sandbox/command_policy.py"),
            ("ShellRunner", "sandbox/shell_runner.py"),
            ("PolicyConfig", "config/policy.yaml"),
        )
        for label, token in checks:
            if any(token in path for path in rel_paths):
                boundaries.append(label)
        return boundaries

    def _collect_entrypoints(self, files: list[Path]) -> list[str]:
        patterns = (
            re.compile(r'if __name__ == ["\']__main__["\']'),
            re.compile(r"FastAPI\("),
            re.compile(r"Typer\("),
            re.compile(r"APIRouter\("),
            re.compile(r"express\("),
        )
        matches: list[str] = []
        for path in files[:40]:
            rel = path.relative_to(self.guard.root).as_posix()
            try:
                content = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            for pattern in patterns:
                if pattern.search(content):
                    matches.append(rel)
                    break
            if len(matches) >= 12:
                break
        return matches

    def _collect_module_edges(self, files: list[Path]) -> list[ModuleEdge]:
        candidates = files[:24]
        edges: list[ModuleEdge] = []
        seen: set[tuple[str, str]] = set()
        for path in candidates:
            rel = path.relative_to(self.guard.root).as_posix()
            try:
                content = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            for target in self._extract_import_targets(content)[:6]:
                if target.startswith(".") or self._should_skip_import_target(target):
                    continue
                edge_key = (rel, target)
                if edge_key in seen:
                    continue
                seen.add(edge_key)
                edges.append(ModuleEdge(source=rel.replace("/", "."), target=target))
                if len(edges) >= 24:
                    return edges
        return edges

    @staticmethod
    def _should_skip_import_target(target: str) -> bool:
        root = target.split(".", 1)[0]
        if root == "__future__":
            return True
        if root in getattr(sys, "stdlib_module_names", set()):
            return True
        if root in {"typing", "pathlib", "json", "re", "time", "subprocess", "tempfile", "collections"}:
            return True
        return False

    @staticmethod
    def _extract_import_targets(content: str) -> list[str]:
        patterns = [
            re.compile(r'^\s*from\s+([A-Za-z0-9_\.]+)\s+import\s+', flags=re.MULTILINE),
            re.compile(r'^\s*import\s+([A-Za-z0-9_\.]+)', flags=re.MULTILINE),
            re.compile(r'^\s*import\s+.*?\s+from\s+[\'"]([^\'"]+)[\'"]', flags=re.MULTILINE),
            re.compile(r'^\s*const\s+.*?=\s+require\([\'"]([^\'"]+)[\'"]\)', flags=re.MULTILINE),
        ]
        found: list[str] = []
        seen: set[str] = set()
        for pattern in patterns:
            for item in pattern.findall(content):
                if item in seen:
                    continue
                seen.add(item)
                found.append(item)
        return found
