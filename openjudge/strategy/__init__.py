# -*- coding: utf-8 -*-
"""Strategy module for evaluation workflows."""

from .base import BaseStrategy
from .direct_strategy import DirectStrategy
from .majority_vote_strategy import MajorityVoteStrategy

__all__ = ["BaseStrategy", "DirectStrategy", "MajorityVoteStrategy"]
