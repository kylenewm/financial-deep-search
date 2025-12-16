"""
Configuration for research features.

Feature flags control which capabilities are enabled.
Discovery is OFF by default — enable with DISCOVERY_ENABLED=true.
"""
import os
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class ResearchConfig:
    """Configuration for research features."""
    
    # Core features (always on)
    xbrl_enabled: bool = True
    drift_enabled: bool = True
    verification_enabled: bool = True
    
    # Discovery feature (off by default)
    discovery_enabled: bool = field(
        default_factory=lambda: os.getenv("DISCOVERY_ENABLED", "false").lower() == "true"
    )
    
    # Which discovery backend to use: "news" (tier 1 sources) or "deep_research" (full web)
    discovery_backend: Literal["news", "deep_research"] = field(
        default_factory=lambda: os.getenv("DISCOVERY_BACKEND", "news")
    )
    
    @classmethod
    def from_env(cls) -> "ResearchConfig":
        """Create config from environment variables."""
        return cls()


# Global default config
default_config = ResearchConfig()



