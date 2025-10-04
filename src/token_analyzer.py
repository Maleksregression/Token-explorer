import os
import sys
import json
import torch
import platform
import math
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
from src.cache import TokenCache
from src.provider import BaseProvider


class TokenAnalyzer:
    """Handles token analysis and prediction logic with provider abstraction"""
    def __init__(self, provider: BaseProvider, cache: TokenCache = None):
        self.provider = provider
        self.cache = cache or TokenCache()

    def get_next_tokens(self, prompt: str, max_tokens: int = 15, temperature: float = 1.0, top_p: float = 1.0) -> List[Tuple[int, str, float]]:
        """Get the next most probable tokens with caching and error handling"""
        cache_key = f"{prompt}_{max_tokens}_{temperature}_{top_p}"
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            return cached_result

        tokens = self.provider.get_next_tokens(prompt, max_tokens, temperature, top_p)

        if tokens:
            self.cache.set(cache_key, tokens)

        return tokens

    def calculate_entropy(self, tokens_data: List[Tuple[int, str, float]]) -> float:
        """Calculate entropy of the probability distribution"""
        if not tokens_data:
            return 0.0

        probs = [prob for _, _, prob in tokens_data]
        total_prob = sum(probs)
        if total_prob == 0:
            return 0.0

        normalized_probs = [p / total_prob for p in probs]
        entropy = -sum(p * math.log2(p) for p in normalized_probs if p > 0)
        return entropy

    def calculate_top_k_entropy(self, tokens_data: List[Tuple[int, str, float]], k: int = 5) -> float:
        """Calculate entropy of top-k probability distribution"""
        if not tokens_data:
            return 0.0
        top_k_tokens = tokens_data[:k]
        return self.calculate_entropy(top_k_tokens)

    def calculate_perplexity(self, tokens_data: List[Tuple[int, str, float]]) -> float:
        """Calculate perplexity of the probability distribution"""
        entropy = self.calculate_entropy(tokens_data)
        return 2 ** entropy

    def get_token_statistics(self, tokens_data: List[Tuple[int, str, float]]) -> Dict[str, Any]:
        """Get comprehensive token statistics"""
        if not tokens_data:
            return {}

        probs = [prob for _, _, prob in tokens_data]
        entropy = self.calculate_entropy(tokens_data)
        perplexity = self.calculate_perplexity(tokens_data)

        max_prob = max(probs) if probs else 0
        min_prob = min(probs) if probs else 0
        avg_prob = sum(probs) / len(probs) if probs else 0

        punctuation_chars = set(".,!?;:\"'()[]{}")
        punctuation_count = sum(
            1 for _, token, _ in tokens_data
            if token and all(c in punctuation_chars for c in token.strip())
        )
        word_count = sum(1 for _, token, _ in tokens_data if token and token.strip().isalpha())
        number_count = sum(1 for _, token, _ in tokens_data if token and token.strip().replace('.', '').isdigit())

        return {
            'entropy': entropy,
            'perplexity': perplexity,
            'max_probability': max_prob,
            'min_probability': min_prob,
            'avg_probability': avg_prob,
            'punctuation_count': punctuation_count,
            'word_count': word_count,
            'number_count': number_count,
            'total_tokens': len(tokens_data)
        }

    def analyze_token_influence(self, original_prompt: str, token_to_test: str,
                                max_tokens: int = 15) -> List[Tuple[int, str, float]]:
        """Analyze how adding a specific token affects the next predictions"""
        modified_prompt = original_prompt + token_to_test
        return self.get_next_tokens(modified_prompt, max_tokens)

    def calculate_mutual_information(self, token1_data: List[Tuple[int, str, float]],
                                     token2_data: List[Tuple[int, str, float]]) -> float:
        """Calculate mutual information between two token probability distributions"""
        ent1 = self.calculate_entropy(token1_data)
        ent2 = self.calculate_entropy(token2_data)

        combined_probs = [prob for _, _, prob in token1_data] + [prob for _, _, prob in token2_data]
        if not combined_probs:
            return 0.0

        total = sum(combined_probs)
        if total == 0:
            return 0.0

        normalized_combined = [p / total for p in combined_probs]
        combined_entropy = -sum(p * math.log2(p) for p in normalized_combined if p > 0)

        mutual_info = max(0, ent1 + ent2 - combined_entropy)
        return mutual_info

    def generate_probability_visualization(self, tokens_data: List[Tuple[int, str, float]], max_display: int = 10) -> str:
        """Generate a text-based visualization of token probabilities"""
        if not tokens_data:
            return "No tokens to visualize"
        
        # Take only the top tokens for visualization
        top_tokens = tokens_data[:max_display]
        
        # Find the max probability for scaling
        max_prob = max([prob for _, _, prob in top_tokens]) if top_tokens else 0.0
        if max_prob == 0.0:
            return "All probabilities are zero"
        
        # Create visualization string
        vis_lines = ["Token Probability Visualization:", "─" * 50]
        
        for i, (token_id, token, prob) in enumerate(top_tokens):
            # Calculate bar length proportional to probability (max 30 chars)
            bar_length = int((prob / max_prob) * 30) if max_prob > 0 else 0
            bar = "█" * bar_length
            
            # Format the token for display (show special characters)
            display_token = repr(token) if token.strip() != token else f"'{token}'"
            
            vis_lines.append(f"{i+1:2d}. {display_token:<10} |{bar:<30}| {prob:.4f}")
        
        return "\n".join(vis_lines)