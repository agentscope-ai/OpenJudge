"""
NLTK Data Setup Script

Download NLTK data packages required for metrics like METEOR.
"""

import nltk
from loguru import logger


def download_nltk_data():
    """Download necessary NLTK data packages"""

    packages = [
        "punkt",  # Tokenizer
        "wordnet",  # WordNet lexicon (required by METEOR)
        "omw-1.4",  # Open Multilingual Wordnet
        "averaged_perceptron_tagger",  # POS tagger
    ]

    logger.info("Starting NLTK data package download...")

    for package in packages:
        try:
            logger.info(f"Downloading {package}...")
            nltk.download(package, quiet=True)
            logger.success(f"✓ {package} download completed")
        except Exception as e:
            logger.error(f"✗ {package} download failed: {e}")

    logger.success("NLTK data packages download completed!")


if __name__ == "__main__":
    download_nltk_data()
