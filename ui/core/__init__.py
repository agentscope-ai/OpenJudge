# -*- coding: utf-8 -*-
"""Core framework for OpenJudge Studio.

This module provides the foundation for the multi-feature UI architecture:
- BaseFeature: Abstract base class for feature modules
- FeatureRegistry: Central registry for feature management
- Navigation: Navigation and routing management
- SessionManager: Session state management utilities
- TaskManager: Background task management
"""

from core.base_feature import BaseFeature
from core.feature_registry import FeatureRegistry
from core.navigation import Navigation
from core.session_manager import FeatureState, SessionManager
from core.task_manager import (
    TaskInfo,
    TaskManager,
    TaskProgress,
    TaskStatus,
    TaskType,
    get_task_manager,
)

__all__ = [
    "BaseFeature",
    "FeatureRegistry",
    "FeatureState",
    "Navigation",
    "SessionManager",
    "TaskInfo",
    "TaskManager",
    "TaskProgress",
    "TaskStatus",
    "TaskType",
    "get_task_manager",
]
