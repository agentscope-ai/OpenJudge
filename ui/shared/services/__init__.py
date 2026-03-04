# -*- coding: utf-8 -*-
"""Shared services for OpenJudge Studio."""

from shared.services.model_factory import create_model
from shared.services.workspace_manager import (
    StorageManager,
    WorkspaceManager,
    get_current_workspace,
    get_current_workspace_path,
    get_storage_manager,
    initialize_workspace_from_url,
    inject_browser_id_loader,
    set_current_workspace,
)

__all__ = [
    "create_model",
    "StorageManager",
    "WorkspaceManager",
    "get_current_workspace",
    "get_current_workspace_path",
    "get_storage_manager",
    "inject_browser_id_loader",
    "initialize_workspace_from_url",
    "set_current_workspace",
]
