# -*- coding: utf-8 -*-
"""TeX package processing and merging."""

import os
import re
import shutil
import tarfile
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union


@dataclass
class TexFile:
    """A single TeX file."""

    path: str
    content: str
    is_main: bool = False


@dataclass
class BibFile:
    """A single BibTeX file with content."""

    path: str
    content: str


@dataclass
class TexPackage:
    """Processed TeX package."""

    main_tex: str
    merged_content: str
    files: List[TexFile] = field(default_factory=list)
    bib_files: List[BibFile] = field(default_factory=list)  # Now includes content
    figure_paths: List[str] = field(default_factory=list)


class TexPackageProcessor:
    """Process TeX packages and merge source files.

    Supports .tar.gz and .zip archives. Automatically finds main.tex
    and resolves \\input{} and \\include{} commands.
    """

    INPUT_PATTERN = re.compile(r"\\input\{([^}]+)\}")
    INCLUDE_PATTERN = re.compile(r"\\include\{([^}]+)\}")
    DOCUMENTCLASS_PATTERN = re.compile(r"\\documentclass")

    TEX_EXTENSIONS = {".tex"}
    BIB_EXTENSIONS = {".bib", ".bbl"}
    FIGURE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".pdf", ".eps", ".svg"}

    def __init__(self):
        self._temp_dir: Optional[str] = None
        self._file_cache: Dict[str, str] = {}

    def process_package(
        self,
        package: Union[str, Path, bytes],
        package_name: Optional[str] = None,
    ) -> TexPackage:
        """Process a TeX package archive.

        Args:
            package: Path to .tar.gz/.zip file or bytes content
            package_name: Name hint when passing bytes

        Returns:
            TexPackage with merged content and bib file contents
        """
        extract_dir = self._extract_archive(package, package_name)

        try:
            tex_files = self._find_files(extract_dir, self.TEX_EXTENSIONS)
            bib_file_paths = self._find_files(extract_dir, self.BIB_EXTENSIONS)
            figure_files = self._find_files(extract_dir, self.FIGURE_EXTENSIONS)

            main_tex_path = self._find_main_tex(tex_files, extract_dir)
            if not main_tex_path:
                raise ValueError("Could not find main .tex file (no \\documentclass found)")

            for tex_path in tex_files:
                self._load_file(tex_path)

            merged_content = self._merge_tex_content(main_tex_path, extract_dir)

            files = [
                TexFile(
                    path=os.path.relpath(p, extract_dir),
                    content=self._file_cache.get(p, ""),
                    is_main=(p == main_tex_path),
                )
                for p in tex_files
            ]

            # Read bib file contents before cleanup
            bib_files = [
                BibFile(
                    path=os.path.relpath(p, extract_dir),
                    content=self._load_file(p),
                )
                for p in bib_file_paths
            ]

            return TexPackage(
                main_tex=os.path.relpath(main_tex_path, extract_dir),
                merged_content=merged_content,
                files=files,
                bib_files=bib_files,
                figure_paths=[os.path.relpath(p, extract_dir) for p in figure_files],
            )
        finally:
            self._cleanup()

    def _extract_archive(
        self,
        package: Union[str, Path, bytes],
        name_hint: Optional[str] = None,
    ) -> str:
        """Extract archive to temporary directory."""
        self._temp_dir = tempfile.mkdtemp(prefix="tex_package_")

        if isinstance(package, bytes):
            ext = ".tar.gz" if name_hint and name_hint.endswith(".tar.gz") else ".zip"
            temp_archive = os.path.join(self._temp_dir, f"archive{ext}")
            with open(temp_archive, "wb") as f:
                f.write(package)
            package = temp_archive

        package_path = Path(package)

        if package_path.suffix == ".gz" or str(package_path).endswith(".tar.gz"):
            with tarfile.open(package_path, "r:gz") as tar:
                tar.extractall(self._temp_dir)
        elif package_path.suffix == ".zip":
            with zipfile.ZipFile(package_path, "r") as zip_ref:
                zip_ref.extractall(self._temp_dir)
        else:
            raise ValueError(f"Unsupported archive format: {package_path.suffix}")

        return self._temp_dir

    def _find_files(self, directory: str, extensions: set) -> List[str]:
        """Find all files with given extensions."""
        files = []
        for root, _, filenames in os.walk(directory):
            for filename in filenames:
                if Path(filename).suffix.lower() in extensions:
                    files.append(os.path.join(root, filename))
        return files

    def _find_main_tex(self, tex_files: List[str], base_dir: str) -> Optional[str]:
        """Find the main .tex file containing \\documentclass."""
        candidates = []

        for tex_path in tex_files:
            content = self._load_file(tex_path)
            if self.DOCUMENTCLASS_PATTERN.search(content):
                score = 0
                filename = os.path.basename(tex_path).lower()
                rel_path = os.path.relpath(tex_path, base_dir)

                if filename in ("main.tex", "paper.tex", "manuscript.tex"):
                    score += 10
                if os.path.dirname(rel_path) == "":
                    score += 5
                score -= len(rel_path.split(os.sep))

                candidates.append((score, tex_path))

        if candidates:
            candidates.sort(reverse=True)
            return candidates[0][1]
        return None

    def _load_file(self, path: str) -> str:
        """Load file content with caching."""
        if path not in self._file_cache:
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    self._file_cache[path] = f.read()
            except (FileNotFoundError, OSError, PermissionError):
                self._file_cache[path] = ""
        return self._file_cache[path]

    def _merge_tex_content(
        self,
        main_path: str,
        base_dir: str,
        visited: Optional[set] = None,
    ) -> str:
        """Recursively merge TeX content by resolving \\input and \\include."""
        if visited is None:
            visited = set()

        if main_path in visited:
            return f"% [Circular reference: {main_path}]\n"

        visited.add(main_path)
        content = self._load_file(main_path)
        current_dir = os.path.dirname(main_path)

        def resolve_include(match: re.Match) -> str:
            include_name = match.group(1).strip()
            if not include_name.endswith(".tex"):
                include_name += ".tex"

            for search_path in [
                os.path.join(current_dir, include_name),
                os.path.join(base_dir, include_name),
            ]:
                if os.path.exists(search_path):
                    included = self._merge_tex_content(search_path, base_dir, visited)
                    return f"% === BEGIN: {include_name} ===\n{included}\n% === END: {include_name} ===\n"

            return f"% [File not found: {include_name}]\n"

        content = self.INPUT_PATTERN.sub(resolve_include, content)
        content = self.INCLUDE_PATTERN.sub(resolve_include, content)

        return content

    def _cleanup(self):
        """Clean up temporary directory."""
        if self._temp_dir and os.path.exists(self._temp_dir):
            shutil.rmtree(self._temp_dir, ignore_errors=True)
        self._temp_dir = None
        self._file_cache.clear()
