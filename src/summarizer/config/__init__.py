"""Configuration loading for summarizer module.

This module provides access to section weights and prompts for document summarization.
The configuration is stored in YAML for easy modification and industry customization.
"""

from functools import lru_cache
from pathlib import Path
from typing import TypedDict

import yaml


class SectionWeight(TypedDict):
    """Type definition for section weight configuration."""

    weight: str  # "high", "medium", or "low"
    max_tokens: int
    description: str
    guidance: str


# Path to the configuration file
CONFIG_DIR = Path(__file__).parent
SECTION_WEIGHTS_PATH = CONFIG_DIR / "section_weights.yaml"


@lru_cache(maxsize=1)
def _load_section_weights() -> dict:
    """Load section weights from YAML file (cached)."""
    with open(SECTION_WEIGHTS_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_section_weights() -> dict[str, SectionWeight]:
    """Get the section weights dictionary.

    Returns:
        Dictionary mapping section IDs to their weight configurations.
        Keys are section identifiers like "Item 1", "Item 1A", "Q&A", etc.
    """
    return _load_section_weights()["sections"]


def get_default_section_weight() -> SectionWeight:
    """Get the default weight for unknown sections.

    Returns:
        Default SectionWeight configuration for sections not in the main dict.
    """
    return _load_section_weights()["default"]


def get_section_weight(section_id: str) -> SectionWeight:
    """Get the weight configuration for a specific section.

    Args:
        section_id: The section identifier (e.g., "Item 1", "Q&A")

    Returns:
        SectionWeight for the section, or default if not found.
    """
    sections = get_section_weights()
    return sections.get(section_id, get_default_section_weight())


# Load on import for backwards compatibility
# These are loaded once and cached via lru_cache
SECTION_WEIGHTS = get_section_weights()
DEFAULT_SECTION_WEIGHT = get_default_section_weight()
