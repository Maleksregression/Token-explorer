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
from src.token_analyzer import TokenAnalyzer


class UIHandler:
    """Handles UI rendering and display"""
    def __init__(self, console: Console, config: TokenExplorerConfig):
        self.console = console
        self.config = config
        self.color_theme = {
            "default": {
                "highlight": "bold black on bright_cyan",
                "rank": "cyan",
                "token_id": "green",
                "token": "yellow",
                "probability": "white",
                "bar": "white",
                "border": "dim",
                "header": "bold magenta",
                "prompt_border": "green",
                "controls_border": "magenta",
                "model_meta_border": "white",
                "token_info_border": "cyan",
                "history_border": "blue",
                "search_border": "bright_magenta",
                "filter_border": "bright_green",
                "entropy_border": "bright_yellow"
            },
            "light": {
                "highlight": "bold black on bright_cyan",
                "rank": "blue",
                "token_id": "green",
                "token": "purple",
                "probability": "black",
                "bar": "white",
                "border": "dim",
                "header": "bold blue",
                "prompt_border": "green",
                "controls_border": "magenta",
                "model_meta_border": "black",
                "token_info_border": "cyan",
                "history_border": "blue",
                "search_border": "bright_magenta",
                "filter_border": "bright_green",
                "entropy_border": "bright_yellow"
            },
            "dark": {
                "highlight": "bold white on bright_yellow",
                "rank": "bright_cyan",
                "token_id": "bright_green",
                "token": "bright_yellow",
                "probability": "bright_white",
                "bar": "bright_white",
                "border": "dim",
                "header": "bold bright_magenta",
                "prompt_border": "bright_green",
                "controls_border": "bright_magenta",
                "model_meta_border": "bright_white",
                "token_info_border": "bright_cyan",
                "history_border": "bright_blue",
                "search_border": "bright_magenta",
                "filter_border": "bright_green",
                "entropy_border": "bright_yellow"
            },
        }

    def get_theme_color(self, element: str) -> str:
        """Get color for specified element based on theme"""
        theme = self.color_theme.get(self.config.theme, self.color_theme["default"])
        return theme.get(element, "white")

    def get_theme_style(self, element: str) -> Style:
        """Get style for specified element based on theme"""
        theme = self.color_theme.get(self.config.theme, self.color_theme["default"])
        color_value = theme.get(element, "white")
        try:
            return Style.parse(color_value)
        except Exception:
            return Style()

    def get_theme_markup(self, element: str) -> str:
        """Return a markup-friendly style string for inline markup usage."""
        theme = self.color_theme.get(self.config.theme, self.color_theme["default"])
        return theme.get(element, "white")

    def build_token_table(self, tokens_data: List[Tuple[int, str, float]], selected_token_idx: int, max_tokens: int = 15,
                          token_filters: Optional[Dict[str, Any]] = None) -> Table:
        """Build rich table for tokens display with filtering capabilities"""
        filtered_tokens = tokens_data
        if token_filters:
            if 'type' in token_filters and token_filters.get('type'):
                ttype = token_filters['type']
                if ttype == 'punctuation':
                    punctuation_chars = set(".,!?;:'()[]{}\"")
                    filtered_tokens = [(tid, token, prob) for tid, token, prob in tokens_data
                                       if token and all(c in punctuation_chars for c in token.strip())]
                elif ttype == 'word':
                    filtered_tokens = [(tid, token, prob) for tid, token, prob in tokens_data
                                       if token and token.strip().isalpha()]
                elif ttype == 'number':
                    filtered_tokens = [(tid, token, prob) for tid, token, prob in tokens_data
                                       if token and token.strip().replace('.', '').isdigit()]
            if 'search' in token_filters and token_filters.get('search'):
                search_term = token_filters['search'].lower()
                filtered_tokens = [(tid, token, prob) for tid, token, prob in filtered_tokens if token and search_term in token.lower()]

        filtered_tokens = filtered_tokens[:max_tokens]

        table = Table(
            title=f"Next Most Probable Tokens ({len(filtered_tokens)}/{len(tokens_data)} shown)",
            box=box.ROUNDED,
            header_style=self.get_theme_style("header"),
            border_style=self.get_theme_style("border")
        )
        table.add_column("Rank", style=self.get_theme_style("rank"))
        table.add_column("Token ID", style=self.get_theme_style("token_id"))
        table.add_column("Token", style=self.get_theme_style("token"))
        table.add_column("Probability", style=self.get_theme_style("probability"))
        table.add_column("Bar", style=self.get_theme_style("bar"))

        if not filtered_tokens:
            table.add_row("-", "-", "(no tokens match filter)", "-", "")
            return table

        max_prob = max([t[2] for t in filtered_tokens]) if filtered_tokens else 0.0
        for i, (token_id, token, prob) in enumerate(filtered_tokens):
            bar_length = int((prob / max_prob) * 20) if max_prob > 0 else 0
            bar = ""

            if bar_length > 0:
                if self.config.theme == "light":
                    intensity = int(150 + (prob / max_prob) * 105)
                    hex_color = f"#{intensity:02x}{intensity:02x}{intensity:02x}"
                else:
                    intensity = int(30 + (prob / max_prob) * 225)
                    hex_color = f"#{intensity:02x}{intensity:02x}00"
                bar = f"[{hex_color}]" + "█" * bar_length + "[/]"

            highlight_markup = self.get_theme_markup("highlight")
            is_selected = (i == selected_token_idx)

            token_display = repr(token)
            punctuation_chars = set(".,!?;:'()[]{}\"")
            if token and all(c in punctuation_chars for c in token.strip()):
                token_display = f"[bold]{token_display}[/bold]"
            elif token and token.strip().isalpha():
                token_display = f"[italic]{token_display}[/italic]"
            elif token and token.strip().replace('.', '').isdigit():
                token_display = f"[underline]{token_display}[/underline]"

            rank_cell = f"[{highlight_markup}] {i+1} [/{highlight_markup}]" if is_selected else str(i+1)
            id_cell = f"[{highlight_markup}] {token_id} [/{highlight_markup}]" if is_selected else str(token_id)
            token_cell = f"[{highlight_markup}] {token_display} [/{highlight_markup}]" if is_selected else token_display
            prob_cell = f"[{highlight_markup}] {prob:.4f} [/{highlight_markup}]" if is_selected else f"{prob:.4f}"

            table.add_row(rank_cell, id_cell, token_cell, prob_cell, bar)
        return table

    def build_prompt_panel(self, prompts: List[str], current_prompt_idx: int, selected_token_idx: int, max_tokens: int = 15,
                           search_query: str = None, filters: Dict[str, Any] = None) -> Group:
        """Build rich panel for prompt display with controls and search/filter options"""
        prompt_text = prompts[current_prompt_idx] if prompts else ""

        extra_info = ""
        if search_query:
            extra_info += f"\n[bold yellow]Search:[/bold yellow] {search_query}"
        if filters:
            filter_strs = [f"{k}: {v}" for k, v in filters.items() if v]
            if filter_strs:
                extra_info += f"\n[bold green]Filters:[/bold green] {' | '.join(filter_strs)}"

        return Panel(
            f"[bold]Prompt {current_prompt_idx+1}/{len(prompts) if prompts else 1}:[/bold]\n\n{prompt_text}{extra_info}",
            border_style=self.get_theme_style("prompt_border"),
            title="Current Prompt",
            expand=True
        )

    def build_controls_panel(self) -> Panel:
        controls = Text()
        controls.append("↑/↓: Navigate tokens\n", style=Style(bold=True, color="cyan"))
        controls.append("←/→ n/p: Prev/Next prompt\n", style=Style(bold=True, color="yellow"))
        controls.append("Enter: Append token | u: Undo\n", style=Style(bold=True, color="green"))
        controls.append("a: Add prompt | m: Switch model\n", style=Style(bold=True, color="magenta"))
        controls.append("i: Inspect token | h: History\n", style=Style(bold=True, color="blue"))
        controls.append("c: Model metadata | s: Save session\n", style=Style(bold=True, color="white"))
        controls.append("f: Filter tokens | /: Search tokens\n", style=Style(bold=True, color="bright_green"))
        controls.append("e: Entropy analysis | t: Toggle theme\n", style=Style(bold=True, color="bright_yellow"))
        controls.append("V/P: Adjust sampling\n", style=Style(bold=True, color="bright_cyan"))
        controls.append("B: Token influence analysis\n", style=Style(bold=True, color="bright_magenta"))
        controls.append("g: Start token exploration game\n", style=Style(bold=True, color="bright_red"))
        controls.append("q: Quit", style=Style(bold=True, color="red"))

        return Panel(controls, title="Controls", border_style=self.get_theme_style("controls_border"), expand=True)

    def build_search_panel(self) -> Panel:
        """Build a panel for token searching"""
        search_info = Text()
        search_info.append("Token Search\n", style=Style(bold=True))
        search_info.append("Enter search term to filter tokens by content\n", style=Style(dim=True))
        search_info.append("Examples: 'the', 'ing', '123', etc.", style=Style(italic=True))

        return Panel(search_info, title="Token Search", border_style=Style(color=self.get_theme_color("search_border")))

    def build_filter_panel(self) -> Panel:
        """Build a panel for token filtering"""
        filter_info = Text()
        filter_info.append("Token Filters\n", style=Style(bold=True))
        filter_info.append("Available filters:\n", style=Style(dim=True))
        filter_info.append("  - word: alphabetic tokens only\n")
        filter_info.append("  - punctuation: punctuation tokens only\n")
        filter_info.append("  - number: numeric tokens only\n")

        return Panel(filter_info, title="Token Filters", border_style=Style(color=self.get_theme_color("filter_border")))

    def build_entropy_panel(self, tokens_data: List[Tuple[int, str, float]], token_analyzer: TokenAnalyzer) -> Panel:
        """Build a visualization panel for token entropy and other statistics"""
        if not tokens_data:
            entropy_text = Text("No token data available", style="dim")
            return Panel(entropy_text, title="Entropy & Statistics", border_style=self.get_theme_style("entropy_border"))

        stats = token_analyzer.get_token_statistics(tokens_data)

        entropy_text = Text()
        entropy_text.append(f"Entropy: {stats['entropy']:.4f}\n", style=Style(bold=True))
        entropy_text.append(f"Perplexity: {stats['perplexity']:.4f}\n", style=Style(bold=True))
        entropy_text.append(f"Max Prob: {stats['max_probability']:.4f}\n")
        entropy_text.append(f"Min Prob: {stats['min_probability']:.4f}\n")
        entropy_text.append(f"Avg Prob: {stats['avg_probability']:.4f}\n", style=Style(dim=True))
        entropy_text.append("---\n", style=Style(dim=True))
        entropy_text.append(f"Words: {stats['word_count']}\n")
        entropy_text.append(f"Punct: {stats['punctuation_count']}\n")
        entropy_text.append(f"Nums: {stats['number_count']}\n")

        return Panel(entropy_text, title="Entropy & Statistics", border_style=self.get_theme_style("entropy_border"))

    def build_interface(self, prompts: List[str], current_prompt_idx: int, tokens_data: List[Tuple[int, str, float]],
                        selected_token_idx: int, max_tokens: int = 15, search_query: str = None,
                        filters: Dict[str, Any] = None, token_analyzer: TokenAnalyzer = None, 
                        temperature: float = 1.0, top_p: float = 1.0) -> Layout:
        """Build the main interface layout with entropy analysis and sampling parameters"""
        layout = Layout()
        layout.split_column(
            Layout(name="top", size=10),
            Layout(name="middle", ratio=3),
            Layout(name="bottom", size=5)
        )

        layout["top"].update(self.build_prompt_panel(prompts, current_prompt_idx, selected_token_idx,
                                                     max_tokens, search_query, filters))
        layout["middle"].split_row(
            Layout(self.build_token_table(tokens_data, selected_token_idx, max_tokens, filters), ratio=5),
            Layout(name="right_col", size=40)
        )
        layout["middle"]["right_col"].split_column(
            Layout(self.build_sampling_controls_panel(temperature, top_p), size=4),
            Layout(self.build_controls_panel(), size=20)
        )
        layout["bottom"].update(self.build_entropy_panel(tokens_data, token_analyzer))

        return layout

    def build_sampling_controls_panel(self, temperature: float = 1.0, top_p: float = 1.0) -> Panel:
        """Build a panel for sampling parameters controls"""
        sampling_info = Text()
        sampling_info.append(f"Temperature: {temperature:.2f}\n", style=Style(bold=True, color="cyan"))
        sampling_info.append(f"Top-p (nucleus): {top_p:.2f}\n\n", style=Style(bold=True, color="yellow"))
        # sampling_info.append("Use [bold green]Shift+V[/bold green] to adjust temperature\n", style=Style(dim=True))
        # sampling_info.append("Use [bold green]Shift+P[/bold green] to adjust top-p\n", style=Style(dim=True))
        
        return Panel(sampling_info, title="Sampling Parameters", border_style=Style(color=self.get_theme_color("controls_border")))

    def build_probability_visualization_panel(self, tokens_data: List[Tuple[int, str, float]], token_analyzer: TokenAnalyzer) -> Panel:
        """Build a panel for probability visualization"""
        if not tokens_data:
            vis_text = Text("No token data available", style="dim")
            return Panel(vis_text, title="Token Probability Visualization", border_style=self.get_theme_style("entropy_border"))

        # Call the token analyzer's visualization method
        visualization = token_analyzer.generate_probability_visualization(tokens_data)
        
        return Panel(visualization, title="Token Probability Visualization", border_style=Style(color=self.get_theme_color("entropy_border")))
