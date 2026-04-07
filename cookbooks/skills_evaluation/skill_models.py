# -*- coding: utf-8 -*-
"""
Skill data models and loader for Agent Skill packages.

Provides the data classes and filesystem loader used to represent and load
Agent Skill packages from disk.  Consumed by :mod:`runner` and other tools in
this cookbook.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from loguru import logger

# ── Constants ──────────────────────────────────────────────────────────────────

SKILL_MD_NAME = "SKILL.md"

_FRONTMATTER_RE = re.compile(r"^---[ \t]*\n(.*?)\n---[ \t]*\n?", re.DOTALL)

_IGNORE_DIRS = {"__pycache__", ".git", "node_modules", ".venv", "venv"}

_FILE_TYPE_MAP = {
    ".py": "python",
    ".sh": "bash",
    ".bash": "bash",
    ".js": "javascript",
    ".ts": "typescript",
    ".md": "markdown",
    ".mdx": "markdown",
}

# ── Skill data models ──────────────────────────────────────────────────────────


@dataclass
class SkillManifest:
    """Parsed YAML frontmatter from SKILL.md.

    Supports the Agent Skills specification format used by OpenAI Codex Skills
    and Cursor Agent Skills.

    Attributes:
        name: Skill identifier (lowercase, alphanumeric + hyphens).
        description: Trigger/description text shown to the agent before loading.
        license: Optional SPDX license identifier.
        compatibility: Optional compatibility string.
        allowed_tools: List of allowed tool names (normalised from comma-separated string).
        metadata: Arbitrary metadata dict from frontmatter.
        raw_yaml: Original YAML string (without ``---`` delimiters).
    """

    name: str
    description: str
    license: Optional[str] = None
    compatibility: Optional[str] = None
    allowed_tools: List[str] = field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = None
    raw_yaml: str = ""

    def __post_init__(self) -> None:
        if self.allowed_tools is None:
            self.allowed_tools = []
        elif isinstance(self.allowed_tools, str):
            parts = [p.strip() for p in self.allowed_tools.split(",")]
            self.allowed_tools = [p for p in parts if p]

    @property
    def short_description(self) -> Optional[str]:
        """Return ``metadata.short-description`` if present (Codex Skills format)."""
        if self.metadata and isinstance(self.metadata, dict):
            return self.metadata.get("short-description")
        return None


@dataclass
class SkillFile:
    """A file within a skill package.

    Attributes:
        path: Absolute filesystem path.
        relative_path: Path relative to the skill's root directory.
        file_type: One of ``python``, ``bash``, ``javascript``, ``typescript``,
            ``markdown``, or ``other``.
        content: UTF-8 text content of the file.
        size_bytes: File size in bytes.
    """

    path: Path
    relative_path: str
    file_type: str
    content: str = ""
    size_bytes: int = 0

    @property
    def is_script(self) -> bool:
        """True for executable script files (Python, Bash, JS, TS)."""
        return self.file_type in ("python", "bash", "javascript", "typescript")


@dataclass
class SkillPackage:
    """Represents a complete Agent Skill package loaded from disk.

    Structure mirrors the Agent Skills specification::

        <skill-name>/
          SKILL.md          ← frontmatter + instructions
          scripts/          ← executable code (optional)
          references/       ← documentation (optional)
          assets/           ← templates / resources (optional)

    Attributes:
        directory: Root directory of the skill package.
        manifest: Parsed YAML frontmatter.
        skill_md_path: Absolute path to SKILL.md.
        instruction_body: SKILL.md content after stripping the YAML frontmatter.
        files: All non-SKILL.md files found under ``directory``.
        referenced_files: Relative paths of files under ``scripts/``,
            ``references/``, and ``assets/`` sub-directories.
    """

    directory: Path
    manifest: SkillManifest
    skill_md_path: Path
    instruction_body: str
    files: List[SkillFile] = field(default_factory=list)
    referenced_files: List[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        return self.manifest.name

    @property
    def description(self) -> str:
        return self.manifest.description

    def get_scripts(self) -> List[SkillFile]:
        """Return all executable script files in the package."""
        return [f for f in self.files if f.is_script]

    def get_markdown_files(self) -> List[SkillFile]:
        """Return all Markdown files in the package (excluding SKILL.md)."""
        return [f for f in self.files if f.file_type == "markdown"]

    # ── Grader input helpers ───────────────────────────────────────────────────

    @property
    def full_skill_md(self) -> str:
        """Reconstruct the full SKILL.md text (YAML frontmatter + instruction body).

        This is the canonical string representation passed to graders that consume
        the entire SKILL.md (threat_analysis, completeness, relevance, structure,
        alignment).
        """
        return f"---\n{self.manifest.raw_yaml}\n---\n{self.instruction_body}"

    @property
    def scripts_text(self) -> str:
        """Concatenate all script files into a single annotated string.

        Each section is prefixed with ``=== <relative_path> ===`` so graders can
        attribute findings to specific files.  Passed as the ``scripts`` argument
        to :class:`SkillThreatAnalysisGrader`.
        """
        parts = [f"=== {sf.relative_path} ===\n{sf.content}" for sf in self.get_scripts()]
        return "\n\n".join(parts)

    @property
    def referenced_files_text(self) -> str:
        """Concatenate non-script referenced files into a single annotated string.

        Covers files under ``scripts/``, ``references/``, and ``assets/`` that are
        *not* executable scripts.  Passed as the ``referenced_files`` argument to
        :class:`SkillThreatAnalysisGrader`.
        """
        ref_files = [f for f in self.files if f.relative_path in self.referenced_files and not f.is_script]
        parts = [f"=== {sf.relative_path} ===\n{sf.content}" for sf in ref_files]
        return "\n\n".join(parts)

    @property
    def script_contents(self) -> List[str]:
        """Return text content of each executable script file."""
        return [sf.content for sf in self.get_scripts()]

    @property
    def reference_contents(self) -> List[str]:
        """Return text content of each non-script referenced file."""
        return [f.content for f in self.files if f.relative_path in self.referenced_files and not f.is_script]


# ── Skill Loader ───────────────────────────────────────────────────────────────


def _guess_file_type(path: Path) -> str:
    return _FILE_TYPE_MAP.get(path.suffix.lower(), "other")


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return ""


class SkillLoader:
    """Loads Agent Skill packages from a directory.

    Supports two directory layouts:

    **Single skill**::

        skills_dir/
          SKILL.md
          scripts/run.py

    **Multi-skill registry** (each subdirectory is a skill)::

        skills_dir/
          code-review/
            SKILL.md
            scripts/review.py
          paper-review/
            SKILL.md
    """

    @classmethod
    def _parse_frontmatter(cls, skill_md_content: str) -> tuple[Optional[SkillManifest], str]:
        """Extract YAML frontmatter and return ``(manifest, instruction_body)``."""
        m = _FRONTMATTER_RE.match(skill_md_content)
        if not m:
            return None, skill_md_content

        raw_yaml = m.group(1)
        instruction_body = skill_md_content[m.end() :]

        try:
            data = yaml.safe_load(raw_yaml) or {}
        except yaml.YAMLError as exc:
            logger.warning(f"YAML parse error in frontmatter: {exc}")
            return None, skill_md_content

        name = str(data.get("name", "")).strip()
        description = str(data.get("description", "")).strip()
        if not name:
            return None, instruction_body

        return (
            SkillManifest(
                name=name,
                description=description,
                license=data.get("license"),
                compatibility=data.get("compatibility"),
                allowed_tools=data.get("allowed-tools") or data.get("allowed_tools") or [],
                metadata=data.get("metadata"),
                raw_yaml=raw_yaml,
            ),
            instruction_body,
        )

    @classmethod
    def _collect_files(cls, skill_dir: Path, skill_md_path: Path) -> tuple[List[SkillFile], List[str]]:
        """Collect all non-SKILL.md files from a skill directory."""
        files: List[SkillFile] = []
        referenced_files: List[str] = []

        _ref_dirs = {"scripts", "references", "assets"}

        for path in sorted(skill_dir.rglob("*")):
            if not path.is_file() or path == skill_md_path:
                continue

            parts = path.relative_to(skill_dir).parts
            if any(p.startswith(".") or p in _IGNORE_DIRS for p in parts):
                continue

            relative = str(path.relative_to(skill_dir))
            file_type = _guess_file_type(path)
            content = _read_text(path)

            files.append(
                SkillFile(
                    path=path,
                    relative_path=relative,
                    file_type=file_type,
                    content=content,
                    size_bytes=path.stat().st_size,
                )
            )

            if parts[0] in _ref_dirs:
                referenced_files.append(relative)

        return files, referenced_files

    @classmethod
    def load_skill(cls, skill_dir: Path) -> Optional[SkillPackage]:
        """Load a single skill from *skill_dir* (must contain ``SKILL.md``).

        Returns ``None`` if ``SKILL.md`` is missing or has no valid frontmatter.
        """
        skill_md_path = skill_dir / SKILL_MD_NAME
        if not skill_md_path.is_file():
            return None

        content = _read_text(skill_md_path)
        manifest, instruction_body = cls._parse_frontmatter(content)
        if manifest is None:
            logger.warning(f"No valid frontmatter in {skill_md_path}; skipping.")
            return None

        files, referenced_files = cls._collect_files(skill_dir, skill_md_path)

        return SkillPackage(
            directory=skill_dir,
            manifest=manifest,
            skill_md_path=skill_md_path,
            instruction_body=instruction_body,
            files=files,
            referenced_files=referenced_files,
        )

    @classmethod
    def load_from_directory(cls, skills_dir: Union[str, Path]) -> List[SkillPackage]:
        """Load all skills from *skills_dir*.

        Args:
            skills_dir: Path to a directory.  If the directory itself contains
                ``SKILL.md`` it is treated as a single-skill directory; otherwise
                each immediate subdirectory is checked for a ``SKILL.md``.

        Returns:
            List of successfully loaded :class:`SkillPackage` objects (may be empty).

        Raises:
            ValueError: If *skills_dir* does not exist or is not a directory.
        """
        skills_dir = Path(skills_dir)
        if not skills_dir.is_dir():
            raise ValueError(f"Not a directory: {skills_dir}")

        if (skills_dir / SKILL_MD_NAME).is_file():
            skill = cls.load_skill(skills_dir)
            return [skill] if skill else []

        skills: List[SkillPackage] = []
        for subdir in sorted(skills_dir.iterdir()):
            if not subdir.is_dir():
                continue
            if any(p in _IGNORE_DIRS for p in subdir.parts):
                continue
            skill = cls.load_skill(subdir)
            if skill:
                skills.append(skill)

        return skills


__all__ = [
    "SKILL_MD_NAME",
    "SkillManifest",
    "SkillFile",
    "SkillPackage",
    "SkillLoader",
    "_FILE_TYPE_MAP",
    "_IGNORE_DIRS",
]
