import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Any


@dataclass
class TokenExplorerConfig:
    """Configuration for the token explorer"""
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    max_tokens: int = 20
    default_prompts: List[str] = None
    theme: str = "default"  # "default", "light", "dark"
    temperature: float = 1.0
    top_p: float = 0.7

    def __post_init__(self):
        if self.default_prompts is None:
            self.default_prompts = [
                "The future of artificial intelligence",
                "In a galaxy far far away",
                "The quick brown fox jumps over",
                "Machine learning is a subset of"
            ]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TokenExplorerConfig':
        return cls(**data)