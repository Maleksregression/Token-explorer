import os
import sys
import json
import torch
import platform
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional, Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM
from rich.console import Console, Group
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt
from rich import box
from rich.text import Text
from rich.layout import Layout
from rich.live import Live
from rich.style import Style
from rich.theme import Theme
import functools
from abc import ABC, abstractmethod
import math
from config import TokenExplorerConfig
from src.provider import HuggingFaceProvider, BaseProvider


class ModelManager:
    """Handles model loading and management using provider abstraction"""
    def __init__(self, config: TokenExplorerConfig):
        self.config = config
        self.provider: BaseProvider = HuggingFaceProvider()  # Default provider
        self.model_name = config.model_name

    def load_model(self, model_name: str = None) -> bool:
        """Load a model with error handling using the provider"""
        if model_name:
            self.model_name = model_name

        return self.provider.load_model(self.model_name)

    def get_model_metadata(self) -> str:
        """Get detailed model metadata using the provider"""
        return self.provider.get_model_metadata()

    def get_next_tokens(self, prompt: str, max_tokens: int = 15, temperature: float = 1.0, top_p: float = 1.0) -> List[Tuple[int, str, float]]:
        """Get the next most probable tokens using the provider"""
        return self.provider.get_next_tokens(prompt, max_tokens, temperature, top_p)

    def get_available_models(self) -> List[str]:
        """Get a list of available models from the current provider"""
        return self.provider.get_available_models()