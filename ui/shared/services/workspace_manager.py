# -*- coding: utf-8 -*-
"""Workspace manager for multi-user isolation without authentication.

Provides user data isolation through:
1. Browser-based anonymous ID (stored in localStorage)
2. Named workspaces that users can create and switch between

This allows multiple users to use the same deployment without
seeing each other's data, while still allowing workspace sharing
when needed (by sharing workspace names).
"""

import re
import secrets
from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st
from loguru import logger

# Session state keys
STATE_BROWSER_ID = "workspace_browser_id"
STATE_CURRENT_WORKSPACE = "workspace_current"
STATE_WORKSPACE_MANAGER = "workspace_manager_instance"

# Base directory for all workspace data
BASE_DIR = Path.home() / ".openjudge_studio"


class WorkspaceManager:
    """Manages workspaces for multi-user isolation.

    Directory structure:
        ~/.openjudge_studio/
        ├── workspaces/
        │   ├── _anonymous_{browser_id}/    # Auto-created for each browser
        │   ├── my_team/                    # User-created named workspace
        │   └── project_alpha/              # Another named workspace
        └── global/                         # Shared resources (if any)
    """

    # Workspace naming rules
    NAME_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")
    MAX_NAME_LENGTH = 64
    MIN_NAME_LENGTH = 3

    # Anonymous workspace prefix
    ANONYMOUS_PREFIX = "_anonymous_"

    def __init__(self, base_dir: Path | None = None):
        """Initialize workspace manager.

        Args:
            base_dir: Base directory for workspaces. Defaults to ~/.openjudge_studio/
        """
        self.base_dir = base_dir or BASE_DIR
        self.workspaces_dir = self.base_dir / "workspaces"
        self.workspaces_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def get_instance(cls) -> "WorkspaceManager":
        """Get or create the singleton instance."""
        if STATE_WORKSPACE_MANAGER not in st.session_state:
            st.session_state[STATE_WORKSPACE_MANAGER] = cls()
        return st.session_state[STATE_WORKSPACE_MANAGER]

    @staticmethod
    def generate_browser_id() -> str:
        """Generate a unique browser ID.

        Returns:
            A random 16-character hex string
        """
        return secrets.token_hex(8)

    @classmethod
    def validate_workspace_name(cls, name: str) -> tuple[bool, str]:
        """Validate a workspace name.

        Args:
            name: The workspace name to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        if not name:
            return False, "workspace.error.name_empty"

        if len(name) < cls.MIN_NAME_LENGTH:
            return False, "workspace.error.name_too_short"

        if len(name) > cls.MAX_NAME_LENGTH:
            return False, "workspace.error.name_too_long"

        if not cls.NAME_PATTERN.match(name):
            return False, "workspace.error.name_invalid_chars"

        if name.startswith(cls.ANONYMOUS_PREFIX):
            return False, "workspace.error.name_reserved"

        return True, ""

    def get_anonymous_workspace_name(self, browser_id: str) -> str:
        """Get the anonymous workspace name for a browser ID.

        Args:
            browser_id: The browser's unique ID

        Returns:
            Anonymous workspace name
        """
        return f"{self.ANONYMOUS_PREFIX}{browser_id}"

    def get_workspace_path(self, workspace_name: str) -> Path:
        """Get the path for a workspace.

        Args:
            workspace_name: Name of the workspace

        Returns:
            Path to the workspace directory
        """
        # Sanitize name to prevent path traversal
        safe_name = re.sub(r"[^a-zA-Z0-9_-]", "", workspace_name)
        return self.workspaces_dir / safe_name

    def ensure_workspace_exists(self, workspace_name: str) -> Path:
        """Ensure a workspace directory exists.

        Args:
            workspace_name: Name of the workspace

        Returns:
            Path to the workspace directory
        """
        workspace_path = self.get_workspace_path(workspace_name)
        workspace_path.mkdir(parents=True, exist_ok=True)

        # Create metadata file if not exists
        meta_file = workspace_path / ".workspace_meta.json"
        if not meta_file.exists():
            import json

            meta = {
                "name": workspace_name,
                "created_at": datetime.now().isoformat(),
                "is_anonymous": workspace_name.startswith(self.ANONYMOUS_PREFIX),
            }
            with open(meta_file, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)

        return workspace_path

    def list_workspaces(self, include_anonymous: bool = False) -> list[dict[str, Any]]:
        """List all available workspaces.

        Args:
            include_anonymous: Whether to include anonymous workspaces

        Returns:
            List of workspace info dictionaries
        """
        workspaces = []

        if not self.workspaces_dir.exists():
            return workspaces

        for ws_dir in self.workspaces_dir.iterdir():
            if not ws_dir.is_dir():
                continue

            name = ws_dir.name
            is_anonymous = name.startswith(self.ANONYMOUS_PREFIX)

            if is_anonymous and not include_anonymous:
                continue

            # Load metadata if available
            meta_file = ws_dir / ".workspace_meta.json"
            meta = {}
            if meta_file.exists():
                import json

                try:
                    with open(meta_file, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                except Exception:
                    pass

            workspaces.append(
                {
                    "name": name,
                    "path": str(ws_dir),
                    "is_anonymous": is_anonymous,
                    "created_at": meta.get("created_at", ""),
                    "display_name": meta.get("display_name", name),
                }
            )

        # Sort: named workspaces first, then by name
        workspaces.sort(key=lambda x: (x["is_anonymous"], x["name"].lower()))
        return workspaces

    def create_workspace(self, name: str, display_name: str | None = None) -> tuple[bool, str]:
        """Create a new named workspace.

        Args:
            name: Workspace name (used in path)
            display_name: Optional display name

        Returns:
            Tuple of (success, error_message)
        """
        # Validate name
        is_valid, error_msg = self.validate_workspace_name(name)
        if not is_valid:
            return False, error_msg

        workspace_path = self.get_workspace_path(name)
        if workspace_path.exists():
            return False, "workspace.error.already_exists"

        # Create workspace
        workspace_path.mkdir(parents=True, exist_ok=True)

        # Create metadata
        import json

        meta = {
            "name": name,
            "display_name": display_name or name,
            "created_at": datetime.now().isoformat(),
            "is_anonymous": False,
        }
        meta_file = workspace_path / ".workspace_meta.json"
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        logger.info(f"Created workspace: {name}")
        return True, ""

    def delete_workspace(self, name: str) -> tuple[bool, str]:
        """Delete a workspace.

        Args:
            name: Workspace name

        Returns:
            Tuple of (success, error_message)
        """
        # Cannot delete anonymous workspaces through this method
        if name.startswith(self.ANONYMOUS_PREFIX):
            return False, "workspace.error.cannot_delete_anonymous"

        workspace_path = self.get_workspace_path(name)
        if not workspace_path.exists():
            return False, "workspace.error.not_found"

        import shutil

        try:
            shutil.rmtree(workspace_path)
            logger.info(f"Deleted workspace: {name}")
            return True, ""
        except Exception as e:
            logger.error(f"Failed to delete workspace {name}: {e}")
            return False, "workspace.error.delete_failed"

    def workspace_exists(self, name: str) -> bool:
        """Check if a workspace exists.

        Args:
            name: Workspace name

        Returns:
            True if workspace exists
        """
        return self.get_workspace_path(name).exists()


def get_current_workspace() -> str:
    """Get the current workspace name from session state.

    Returns:
        Current workspace name
    """
    return st.session_state.get(STATE_CURRENT_WORKSPACE, "")


def get_current_workspace_path() -> Path:
    """Get the current workspace path.

    Returns:
        Path to current workspace directory
    """
    manager = WorkspaceManager.get_instance()
    workspace_name = get_current_workspace()

    if not workspace_name:
        # Fallback to anonymous workspace
        browser_id = st.session_state.get(STATE_BROWSER_ID, "default")
        workspace_name = manager.get_anonymous_workspace_name(browser_id)

    return manager.ensure_workspace_exists(workspace_name)


def set_current_workspace(workspace_name: str) -> None:
    """Set the current workspace.

    Args:
        workspace_name: Name of workspace to switch to
    """
    st.session_state[STATE_CURRENT_WORKSPACE] = workspace_name


class StorageManager:
    """Manages storage limits and cleanup for workspaces.

    Provides:
    - Storage quota enforcement
    - Automatic cleanup of old data
    - Usage statistics
    """

    # Default limits
    DEFAULT_MAX_STORAGE_MB = 500  # 500 MB per workspace
    DEFAULT_MAX_HISTORY_ITEMS = 100  # Max history items per feature
    DEFAULT_RETENTION_DAYS = 30  # Keep data for 30 days

    def __init__(self, workspace_path: Path):
        """Initialize storage manager for a workspace.

        Args:
            workspace_path: Path to the workspace directory
        """
        self.workspace_path = workspace_path

    def get_storage_usage(self) -> dict[str, Any]:
        """Get storage usage statistics for the workspace.

        Returns:
            Dictionary with usage statistics
        """
        total_size = 0
        file_count = 0
        dir_sizes: dict[str, int] = {}

        if not self.workspace_path.exists():
            return {
                "total_size_bytes": 0,
                "total_size_mb": 0,
                "file_count": 0,
                "directories": {},
            }

        for item in self.workspace_path.rglob("*"):
            if item.is_file():
                size = item.stat().st_size
                total_size += size
                file_count += 1

                # Track by top-level directory
                try:
                    rel_path = item.relative_to(self.workspace_path)
                    top_dir = rel_path.parts[0] if rel_path.parts else "root"
                    dir_sizes[top_dir] = dir_sizes.get(top_dir, 0) + size
                except ValueError:
                    pass

        return {
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "file_count": file_count,
            "directories": {k: round(v / (1024 * 1024), 2) for k, v in dir_sizes.items()},
        }

    def cleanup_old_data(
        self,
        retention_days: int = DEFAULT_RETENTION_DAYS,
        max_items: int = DEFAULT_MAX_HISTORY_ITEMS,
    ) -> dict[str, int]:
        """Clean up old data from the workspace.

        Args:
            retention_days: Keep data newer than this many days
            max_items: Maximum items to keep per directory

        Returns:
            Dictionary with cleanup statistics
        """
        import shutil
        from datetime import timedelta

        cutoff_time = datetime.now() - timedelta(days=retention_days)
        stats = {"deleted_dirs": 0, "deleted_files": 0, "freed_bytes": 0}

        # Directories to clean
        cleanup_dirs = ["evaluations", "batch_evaluations", "rubrics"]

        for dir_name in cleanup_dirs:
            dir_path = self.workspace_path / dir_name
            if not dir_path.exists():
                continue

            # Get all subdirectories with their modification times
            subdirs = []
            for subdir in dir_path.iterdir():
                if subdir.is_dir():
                    try:
                        mtime = datetime.fromtimestamp(subdir.stat().st_mtime)
                        size = sum(f.stat().st_size for f in subdir.rglob("*") if f.is_file())
                        subdirs.append((subdir, mtime, size))
                    except Exception:
                        continue

            # Sort by modification time (newest first)
            subdirs.sort(key=lambda x: x[1], reverse=True)

            # Delete old items beyond retention period and max count
            for i, (subdir, mtime, size) in enumerate(subdirs):
                should_delete = False

                # Delete if beyond max items
                if i >= max_items:
                    should_delete = True

                # Delete if older than retention period (but keep at least 10 items)
                if mtime < cutoff_time and i >= 10:
                    should_delete = True

                if should_delete:
                    try:
                        shutil.rmtree(subdir)
                        stats["deleted_dirs"] += 1
                        stats["freed_bytes"] += size
                        logger.info(f"Cleaned up old data: {subdir.name}")
                    except Exception as e:
                        logger.warning(f"Failed to delete {subdir}: {e}")

        stats["freed_mb"] = round(stats["freed_bytes"] / (1024 * 1024), 2)
        return stats

    def enforce_quota(self, max_mb: int = DEFAULT_MAX_STORAGE_MB) -> bool:
        """Check if workspace is within storage quota.

        Args:
            max_mb: Maximum storage in MB

        Returns:
            True if within quota, False if exceeded
        """
        usage = self.get_storage_usage()
        return usage["total_size_mb"] <= max_mb

    def auto_cleanup_if_needed(
        self,
        max_mb: int = DEFAULT_MAX_STORAGE_MB,
        retention_days: int = DEFAULT_RETENTION_DAYS,
    ) -> dict[str, Any] | None:
        """Automatically cleanup if storage quota is exceeded.

        Args:
            max_mb: Maximum storage in MB
            retention_days: Retention period for cleanup

        Returns:
            Cleanup stats if cleanup was performed, None otherwise
        """
        usage = self.get_storage_usage()

        if usage["total_size_mb"] > max_mb:
            logger.info(
                f"Workspace storage ({usage['total_size_mb']}MB) exceeds quota ({max_mb}MB), " "performing cleanup..."
            )

            # Progressively more aggressive cleanup
            stats = self.cleanup_old_data(retention_days=retention_days, max_items=50)

            # If still over quota, be more aggressive
            new_usage = self.get_storage_usage()
            if new_usage["total_size_mb"] > max_mb:
                more_stats = self.cleanup_old_data(retention_days=7, max_items=20)
                stats["deleted_dirs"] += more_stats["deleted_dirs"]
                stats["freed_bytes"] += more_stats["freed_bytes"]
                stats["freed_mb"] = round(stats["freed_bytes"] / (1024 * 1024), 2)

            return stats

        return None


def get_storage_manager() -> StorageManager:
    """Get storage manager for the current workspace.

    Returns:
        StorageManager instance for current workspace
    """
    workspace_path = get_current_workspace_path()
    return StorageManager(workspace_path)


def inject_browser_id_loader() -> None:
    """Inject JavaScript to load/generate browser ID from localStorage.

    This should be called early in the app to set up browser identification.
    """
    # JavaScript to handle browser ID
    js_code = """
    <script>
    (function() {
        const STORAGE_KEY = 'openjudge_browser_id';

        // Get or generate browser ID
        let browserId = localStorage.getItem(STORAGE_KEY);
        if (!browserId) {
            // Generate a random ID
            browserId = Array.from(crypto.getRandomValues(new Uint8Array(8)))
                .map(b => b.toString(16).padStart(2, '0'))
                .join('');
            localStorage.setItem(STORAGE_KEY, browserId);
        }

        // Send to Streamlit via query params (one-time setup)
        const urlParams = new URLSearchParams(window.location.search);
        const currentId = urlParams.get('_bid');

        if (currentId !== browserId) {
            urlParams.set('_bid', browserId);
            // Use replaceState to avoid adding to history
            const newUrl = window.location.pathname + '?' + urlParams.toString();
            window.history.replaceState({}, '', newUrl);
            // Reload to let Streamlit pick up the param
            if (!currentId) {
                window.location.reload();
            }
        }
    })();
    </script>
    """
    st.components.v1.html(js_code, height=0)


def initialize_workspace_from_url() -> None:
    """Initialize workspace from URL parameters.

    Reads browser ID from URL and sets up the workspace.
    Should be called after inject_browser_id_loader().
    """
    # Get browser ID from query params
    query_params = st.query_params
    browser_id = query_params.get("_bid", "")

    if browser_id:
        st.session_state[STATE_BROWSER_ID] = browser_id

        # If no workspace is set, use anonymous workspace
        if STATE_CURRENT_WORKSPACE not in st.session_state:
            manager = WorkspaceManager.get_instance()
            anonymous_ws = manager.get_anonymous_workspace_name(browser_id)
            st.session_state[STATE_CURRENT_WORKSPACE] = anonymous_ws
            # Ensure it exists
            manager.ensure_workspace_exists(anonymous_ws)
    elif STATE_BROWSER_ID not in st.session_state:
        # Fallback: generate server-side ID (less ideal but works)
        browser_id = WorkspaceManager.generate_browser_id()
        st.session_state[STATE_BROWSER_ID] = browser_id

        manager = WorkspaceManager.get_instance()
        anonymous_ws = manager.get_anonymous_workspace_name(browser_id)
        st.session_state[STATE_CURRENT_WORKSPACE] = anonymous_ws
        manager.ensure_workspace_exists(anonymous_ws)
