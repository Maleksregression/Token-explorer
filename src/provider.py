import os
import sys
import json
import torch
import platform
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional, Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM
from rich.console import Console
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


class BaseProvider(ABC):
    """Abstract base class for different AI model providers"""

    @abstractmethod
    def load_model(self, model_name: str) -> bool:
        """Load a model from the provider"""
        pass

    @abstractmethod
    def get_next_tokens(self, prompt: str, max_tokens: int = 15) -> List[Tuple[int, str, float]]:
        """Get the next most probable tokens from the model"""
        pass

    @abstractmethod
    def get_model_metadata(self) -> str:
        """Get detailed model metadata"""
        pass

    @abstractmethod
    def get_available_models(self) -> List[str]:
        """Get a list of available models from this provider"""
        pass


class HuggingFaceProvider(BaseProvider):
    """Provider implementation for Hugging Face models"""

    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.model_name = None
        self.console = Console()

    def load_model(self, model_name: str) -> bool:
        """Load a HuggingFace model with error handling"""
        self.console.print(Panel(f"[bold blue]Loading model: {model_name}[/bold blue]", expand=False))

        try:
            # Clear previous model if exists
            if self.model is not None:
                del self.model
            if self.tokenizer is not None:
                del self.tokenizer

            # Load new model
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            self.model.eval()
            self.model_name = model_name

            self.console.print("[green]Model loaded successfully![/green]\n")
            return True

        except Exception as e:
            self.console.print(f"[red]Error loading model: {str(e)}[/red]")
            # Set to None to avoid using a partially loaded model
            self.tokenizer = None
            self.model = None
            return False

    def get_next_tokens(self, prompt: str, max_tokens: int = 15, temperature: float = 1.0, top_p: float = 1.0) -> List[Tuple[int, str, float]]:
        """Get the next most probable tokens from the HuggingFace model with sampling parameters"""
        if not self.model or not self.tokenizer:
            return []

        try:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            with torch.no_grad():
                outputs = self.model(**inputs, return_dict=True)
                logits = outputs.logits[:, -1, :]
                
                # Apply temperature scaling
                if temperature != 1.0:
                    logits = logits / temperature
                
                # Compute probabilities
                probs = torch.softmax(logits, dim=-1)
                
                # Apply top-p (nucleus) sampling if top_p is less than 1.0
                if top_p < 1.0:
                    # Sort probabilities in descending order
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
                    
                    # Calculate cumulative probabilities
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                    
                    # Create mask for tokens in top-p
                    mask = cumulative_probs <= top_p
                    
                    # Zero out probabilities outside top-p
                    sorted_masked_probs = torch.where(mask, sorted_probs, torch.zeros_like(sorted_probs))
                    
                    # Renormalize probabilities
                    renormalized_probs = sorted_masked_probs / sorted_masked_probs.sum(dim=-1, keepdim=True)
                    
                    # Get top-k tokens based on renormalized probabilities
                    top_k_probs, top_k_indices = torch.topk(renormalized_probs, max_tokens)
                else:
                    # Standard top-k without top-p
                    top_k_probs, top_k_indices = torch.topk(probs, max_tokens)

                tokens: List[Tuple[int, str, float]] = []
                for i in range(max_tokens):
                    token_id = top_k_indices[0][i].item()
                    token = self.tokenizer.decode([token_id])
                    prob = float(top_k_probs[0][i].item())
                    tokens.append((token_id, token, prob))

            return tokens

        except Exception as e:
            console = Console()
            console.print(f"[red]Error getting tokens: {str(e)}[/red]")
            return []

    def get_model_metadata(self) -> str:
        """Get detailed HuggingFace model metadata"""
        if not self.model:
            return "no model loaded"

        try:
            config = self.model.config
            num_params = sum(p.numel() for p in self.model.parameters())
            num_params_str = (f"{num_params/1e9:.2f}B" if num_params > 1e9
                              else f"{num_params/1e6:.2f}M")
            vocab_size = getattr(config, "vocab_size", "?")
            max_positions = getattr(config, "max_position_embeddings", "?")
            num_layers = getattr(config, "num_hidden_layers", "?")

            metadata = (
                f"[bold]Model:[/bold] {self.model_name}\\n"
                f"[bold]Vocabulary size:[/bold] {vocab_size}\\n"
                f"[bold]Max position embeddings:[/bold] {max_positions}\\n"
                f"[bold]Parameters:[/bold] {num_params_str}\\n"
                f"[bold]Layers:[/bold] {num_layers}"
            )
            return metadata
        except Exception as e:
            return f"Error reading metadata: {str(e)}"

    def get_available_models(self) -> List[str]:
        """Get a list of commonly used models (this is a simplified implementation;
        in a real application, this would connect to the HuggingFace API)"""
        return [
            "Qwen/Qwen2.5-0.5B-Instruct",
            
        ]