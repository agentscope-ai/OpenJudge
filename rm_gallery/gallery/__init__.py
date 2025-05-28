
from _del.registry import GalleryRegistry
from gallery.rm.helpfulness import HelpfulnessReward


GalleryRegistry.register("helpfulness", HelpfulnessReward)