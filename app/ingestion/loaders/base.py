"""Base document loader interface."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, List
from pathlib import Path

@dataclass
class LoadedDocument:
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    source_path: str = ""

class BaseLoader(ABC):
    """Abstract base document loader."""

    @abstractmethod
    def load(self, file_path: str | Path) -> LoadedDocument:
        """Load and extract text from a file."""
        pass
