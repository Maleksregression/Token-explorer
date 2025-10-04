import os
import sys
import json
import torch
import platform
from dataclasses import asdict
from typing import List, Tuple, Optional, Dict, Any
from config import TokenExplorerConfig
from src.cache import TokenCache
from src.model_manager import ModelManager
from src.token_analyzer import TokenAnalyzer
from src.ui_handler import UIHandler
from src.input_handler import InputHandler
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


class TokenExplorer:
    def __init__(self, config: TokenExplorerConfig = None):
        self.config = config or TokenExplorerConfig()
        self.console = Console()
        self.model_manager = ModelManager(self.config)
        self.token_cache = TokenCache()
        # Change the TokenAnalyzer to use the provider directly instead of ModelManager
        self.token_analyzer = TokenAnalyzer(self.model_manager.provider, self.token_cache)
        self.ui_handler = UIHandler(self.console, self.config)
        self.input_handler = InputHandler()

        # Initialize state
        self.prompts = self.config.default_prompts.copy()
        self.current_prompt_idx = 0
        self.tokens_data: List[Tuple[int, str, float]] = []
        self.selected_token_idx = 0
        self.max_tokens = self.config.max_tokens
        self.history: List[str] = []  # stores prompt history
        self.undo_stack: List[str] = []  # for undo functionality
        # Add new state variables for search and filters
        self.search_query: Optional[str] = None
        self.filters: Dict[str, Any] = {}
        self.filtered_tokens_data: List[Tuple[int, str, float]] = []
        # Add sampling parameters
        self.temperature = self.config.temperature
        self.top_p = self.config.top_p
        # Add game state variables
        self.game_active = False
        self.game_current_score = 0
        self.game_target_prompt = ""
        self.game_user_prompt = ""
        self.game_max_turns = 10
        self.game_current_turn = 0

    def analyze_token_influence(self, live):
        """Analyze how adding specific tokens affects the next predictions"""
        live.stop()
        self.console.print("[bold blue]Token Influence Analysis[/bold blue]")
        
        if not self.tokens_data:
            self.console.print("[red]No tokens to analyze![/red]")
            input("Press Enter to continue...")
            live.start()
            return

        # Show the current prompt and the top possible next tokens
        self.console.print(f"[bold]Current prompt:[/bold] {self.prompts[self.current_prompt_idx]}")
        self.console.print("\n[bold]Top 5 next tokens from current prompt:[/bold]")
        for i, (token_id, token, prob) in enumerate(self.tokens_data[:5]):
            self.console.print(f"  {i+1}. {repr(token)} (ID: {token_id}, Prob: {prob:.4f})")

        # Allow user to select a token to test its influence
        try:
            token_choice = Prompt.ask(
                "\n[bold]Select a token to test its influence (1-5) or enter custom token:[/bold]",
                default="1"
            )
            
            if token_choice.isdigit() and 1 <= int(token_choice) <= 5:
                selected_token = self.tokens_data[int(token_choice) - 1][1]
            else:
                selected_token = token_choice  # Use custom input as token
            
            # Get predictions after adding the selected token
            influence_tokens = self.token_analyzer.analyze_token_influence(
                self.prompts[self.current_prompt_idx], 
                selected_token
            )
            
            self.console.print(f"\n[bold]Top 5 next tokens after adding '{selected_token}':[/bold]")
            for i, (token_id, token, prob) in enumerate(influence_tokens[:5]):
                self.console.print(f"  {i+1}. {repr(token)} (ID: {token_id}, Prob: {prob:.4f})")
            
            # Show comparison
            self.console.print("\n[bold]Comparison:[/bold]")
            self.console.print("[underline]Original predictions:[/underline]")
            for i, (token_id, token, prob) in enumerate(self.tokens_data[:3]):
                self.console.print(f"  {i+1}. {repr(token)} (Prob: {prob:.4f})")
            
            self.console.print(f"[underline]After adding '{selected_token}':[/underline]")
            for i, (token_id, token, prob) in enumerate(influence_tokens[:3]):
                self.console.print(f"  {i+1}. {repr(token)} (Prob: {prob:.4f})")
                
        except Exception as e:
            self.console.print(f"[red]Error during influence analysis: {str(e)}[/red]")
        
        input("\nPress Enter to continue...")
        live.start()

    def start_token_game(self, live):
        """Start the token exploration game"""
        live.stop()
        self.console.print("[bold blue]Token Exploration Game[/bold blue]")
        self.console.print("Try to construct a specific target prompt by selecting tokens!")
        
        # Get target prompt from user
        self.game_target_prompt = Prompt.ask("[bold]Enter the target prompt you want to reach[/bold]")
        if not self.game_target_prompt:
            self.console.print("[yellow]No target provided, cancelling game.[/yellow]")
            live.start()
            return
            
        self.game_user_prompt = self.prompts[self.current_prompt_idx]  # Start from current prompt
        self.game_current_score = 0
        self.game_current_turn = 0
        self.game_active = True
        
        self.console.print(f"[bold]Target:[/bold] {self.game_target_prompt}")
        self.console.print(f"[bold]Starting prompt:[/bold] {self.game_user_prompt}")
        self.console.print(f"[bold]You have {self.game_max_turns} turns to get as close as possible![/bold]")
        
        input("\nPress Enter to start the game...")
        live.start()

    def play_game_turn(self, live):
        """Play a single turn of the token game"""
        if not self.game_active:
            return

        self.game_current_turn += 1
        self.console.print(f"\n[bold]Turn {self.game_current_turn}/{self.game_max_turns}[/bold]")
        self.console.print(f"[bold]Current prompt:[/bold] {self.game_user_prompt}")

        # Calculate and display score based on similarity to target
        similarity = self.calculate_similarity(self.game_user_prompt, self.game_target_prompt)
        self.game_current_score = int(similarity * 100)  # Convert to percentage
        self.console.print(f"[bold]Current Score:[/bold] {self.game_current_score}/100")

        # Get possible next tokens
        game_tokens_data = self.token_analyzer.get_next_tokens(
            self.game_user_prompt, 
            self.max_tokens, 
            self.temperature, 
            self.top_p
        )
        
        if not game_tokens_data:
            self.console.print("[red]No tokens available for this prompt![/red]")
            self.end_game(live)
            return

        # Show possible tokens
        self.console.print("\n[bold]Choose your next token:[/bold]")
        for i, (token_id, token, prob) in enumerate(game_tokens_data[:10]):
            self.console.print(f"  {i+1}. {repr(token)} (Prob: {prob:.4f})")

        try:
            choice = Prompt.ask("[bold]Select token (1-10) or 'end' to finish early[/bold]", default="1")
            if choice.lower() == 'end':
                self.end_game(live)
                return

            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(game_tokens_data):
                selected_token = game_tokens_data[choice_idx][1]
                self.game_user_prompt += selected_token
                
                self.console.print(f"[bold]Selected:[/bold] {repr(selected_token)}")
                self.console.print(f"[bold]New prompt:[/bold] {self.game_user_prompt}")
            else:
                self.console.print("[red]Invalid selection![/red]")
        except ValueError:
            self.console.print("[red]Invalid input![/red]")

        # Check if game is over
        if self.game_current_turn >= self.game_max_turns:
            self.end_game(live)

        input("\nPress Enter to continue...")
        live.start()

    def calculate_similarity(self, str1, str2):
        """Simple similarity calculation based on common characters"""
        if not str1 and not str2:
            return 1.0
        if not str1 or not str2:
            return 0.0
        
        # Use a simple longest common subsequence approach
        # We'll use the length of common characters as a percentage of the longer string
        len1, len2 = len(str1), len(str2)
        max_len = max(len1, len2)
        
        if max_len == 0:
            return 1.0
        
        # Compute basic similarity
        common_chars = set(str1) & set(str2)
        similarity = len(common_chars) / len(set(str1) | set(str2)) if set(str1) | set(str2) else 1.0
        
        return similarity

    def end_game(self, live):
        """End the token exploration game and show results"""
        self.console.print("\n[bold blue]Game Over![/bold blue]")
        self.console.print(f"[bold]Target prompt:[/bold] {self.game_target_prompt}")
        self.console.print(f"[bold]Your prompt:[/bold] {self.game_user_prompt}")
        self.console.print(f"[bold]Final Score:[/bold] {self.game_current_score}/100")
        
        # Calculate final similarity score
        final_similarity = self.calculate_similarity(self.game_user_prompt, self.game_target_prompt)
        final_score = int(final_similarity * 100)
        self.console.print(f"[bold]Final Similarity Score:[/bold] {final_score}/100")
        
        input("\nPress Enter to continue...")
        self.game_active = False
        live.start()

    def toggle_theme(self):
        """Toggle between available themes"""
        themes = ["default", "light", "dark"]
        current_index = themes.index(self.config.theme) if self.config.theme in themes else 0
        next_index = (current_index + 1) % len(themes)
        self.config.theme = themes[next_index]
        self.ui_handler = UIHandler(self.console, self.config)  # Refresh UI handler with new theme

    def run(self):
        if not self.model_manager.load_model():
            return
        self.tokens_data = self.token_analyzer.get_next_tokens(
            self.prompts[self.current_prompt_idx], 
            self.max_tokens, 
            self.temperature, 
            self.top_p
        )

        with Live(self.ui_handler.build_interface(
            self.prompts,
            self.current_prompt_idx,
            self.tokens_data,
            self.selected_token_idx,
            self.max_tokens,
            self.search_query,
            self.filters,
            self.token_analyzer
        ), refresh_per_second=10, screen=True) as live:
            while True:
                try:
                    key = self.input_handler.get_key()
                    if key == 'q':
                        live.stop()
                        self.console.print("[bold green]Goodbye![/bold green]")
                        break
                    elif key in ('n', 'right'):
                        if self.current_prompt_idx < len(self.prompts) - 1:
                            self.current_prompt_idx += 1
                            self.selected_token_idx = 0
                            self.tokens_data = self.token_analyzer.get_next_tokens(
                                self.prompts[self.current_prompt_idx], 
                                self.max_tokens, 
                                self.temperature, 
                                self.top_p
                            )
                    elif key == 'left':
                        if self.current_prompt_idx > 0:
                            self.current_prompt_idx -= 1
                            self.selected_token_idx = 0
                            self.tokens_data = self.token_analyzer.get_next_tokens(
                                self.prompts[self.current_prompt_idx], 
                                self.max_tokens, 
                                self.temperature, 
                                self.top_p
                            )
                    elif key == 'a':
                        live.stop()
                        new_prompt = Prompt.ask("[bold]Enter new prompt[/bold]")
                        if new_prompt:
                            self.prompts.append(new_prompt)
                            self.current_prompt_idx = len(self.prompts) - 1
                            self.tokens_data = self.token_analyzer.get_next_tokens(
                                new_prompt, 
                                self.max_tokens, 
                                self.temperature, 
                                self.top_p
                            )
                        live.start()
                    elif key == 'm':
                        live.stop()
                        # Show available models to user
                        available_models = self.model_manager.get_available_models()
                        self.console.print(Panel(
                            "\n".join(available_models),
                            title="Available Models",
                            border_style=self.ui_handler.get_theme_style("model_meta_border")
                        ))
                        new_model = Prompt.ask("[bold]Enter HuggingFace model name[/bold]", default=self.model_manager.model_name)
                        if new_model:
                            if self.model_manager.load_model(new_model):
                                self.tokens_data = self.token_analyzer.get_next_tokens(
                                    self.prompts[self.current_prompt_idx], 
                                    self.max_tokens, 
                                    self.temperature, 
                                    self.top_p
                                )
                        live.start()
                    elif key == 'c':
                        live.stop()
                        metadata = self.model_manager.get_model_metadata()
                        self.console.print(Panel(
                            metadata,
                            title="Model Metadata",
                            border_style=self.ui_handler.get_theme_style("model_meta_border")
                        ))
                        input("Press Enter to continue...")
                        live.start()
                    elif key in ('\r', '\n'):
                        if self.tokens_data and 0 <= self.selected_token_idx < len(self.tokens_data):
                            token = self.tokens_data[self.selected_token_idx][1]
                            self.undo_stack.append(self.prompts[self.current_prompt_idx])
                            self.prompts[self.current_prompt_idx] += token
                            self.tokens_data = self.token_analyzer.get_next_tokens(
                                self.prompts[self.current_prompt_idx], 
                                self.max_tokens, 
                                self.temperature, 
                                self.top_p
                            )
                            self.selected_token_idx = 0
                            self.history.append(self.prompts[self.current_prompt_idx])
                    elif key == 'u':
                        if self.undo_stack:
                            self.prompts[self.current_prompt_idx] = self.undo_stack.pop()
                            self.tokens_data = self.token_analyzer.get_next_tokens(
                                self.prompts[self.current_prompt_idx], 
                                self.max_tokens, 
                                self.temperature, 
                                self.top_p
                            )
                    elif key == 'i':
                        if self.tokens_data:
                            token_id, token, prob = self.tokens_data[self.selected_token_idx]
                            # Use the model manager's tokenizer to encode
                            if hasattr(self.model_manager.provider, 'tokenizer') and self.model_manager.provider.tokenizer is not None:
                                enc = self.model_manager.provider.tokenizer.encode(token, add_special_tokens=False)
                            else:
                                enc = []
                            info = f"[bold]Token:[/bold] {repr(token)}\n[bold]ID:[/bold] {token_id}\n[bold]Bytes:[/bold] {list(token.encode())}\n[bold]BPE IDs:[/bold] {enc}\n[bold]Prob:[/bold] {prob:.4f}"
                            live.stop()
                            self.console.print(Panel(
                                info,
                                title="Token Info",
                                border_style=self.ui_handler.get_theme_style("token_info_border")
                            ))
                            input("Press Enter to continue...")
                            live.start()
                    elif key == 'h':
                        live.stop()
                        history_text = "\n".join([f"{i+1}. {p}" for i, p in enumerate(self.history)]) or "(empty)"
                        self.console.print(Panel(
                            history_text,
                            title="History",
                            border_style=self.ui_handler.get_theme_style("history_border")
                        ))
                        input("Press Enter to continue...")
                        live.start()
                    elif key == 's':
                        live.stop()
                        filename = Prompt.ask("Enter filename to save", default="session.txt")
                        with open(filename, "w", encoding="utf-8") as f:
                            f.write("\n".join(self.prompts))
                        self.console.print(f"[green]Session saved to {filename}![/green]")
                        input("Press Enter to continue...")
                        live.start()
                    elif key == 't':  # New: Toggle theme
                        self.toggle_theme()
                    elif key == '/':  # New: Search tokens
                        live.stop()
                        search_query = Prompt.ask("[bold]Enter search term (or press Enter to clear)[/bold]")
                        if search_query:
                            self.search_query = search_query
                            if 'search' not in self.filters:
                                self.filters['search'] = None
                            self.filters['search'] = search_query
                        else:
                            self.search_query = None
                            if 'search' in self.filters:
                                del self.filters['search']
                        live.start()
                    elif key == 'f':  # New: Filter tokens
                        live.stop()
                        self.console.print(self.ui_handler.build_filter_panel())
                        filter_type = Prompt.ask("[bold]Enter filter type (word/punctuation/number/all) or press Enter to clear[/bold]")
                        if filter_type and filter_type.lower() in ['word', 'punctuation', 'number']:
                            if 'type' not in self.filters:
                                self.filters['type'] = None
                            self.filters['type'] = filter_type.lower()
                        else:
                            # clear filter if 'all' or empty or unknown
                            if 'type' in self.filters:
                                del self.filters['type']
                        live.start()
                    elif key == 'e':  # New: Detailed entropy analysis
                        live.stop()
                        stats = self.token_analyzer.get_token_statistics(self.tokens_data)
                        entropy_info = ""
                        entropy_info += f"[bold]Entropy:[/bold] {stats.get('entropy', 0.0):.4f}\n"
                        entropy_info += f"[bold]Perplexity:[/bold] {stats.get('perplexity', 0.0):.4f}\n"
                        entropy_info += f"[bold]Max Prob:[/bold] {stats.get('max_probability', 0.0):.4f}\n"
                        entropy_info += f"[bold]Min Prob:[/bold] {stats.get('min_probability', 0.0):.4f}\n"
                        entropy_info += f"[bold]Avg Prob:[/bold] {stats.get('avg_probability', 0.0):.4f}\n"
                        entropy_info += f"[bold]Token Types:[/bold] {stats.get('word_count',0)} words, {stats.get('punctuation_count',0)} punct, {stats.get('number_count',0)} nums"

                        self.console.print(Panel(
                            entropy_info,
                            title="Detailed Entropy Analysis",
                            border_style=self.ui_handler.get_theme_style("entropy_border")
                        ))
                        input("Press Enter to continue...")
                        live.start()
                    elif key in ('v', 'V'):  # Adjust temperature
                        live.stop()
                        try:
                            new_temp = Prompt.ask(
                                f"[bold]Current temperature: {self.temperature:.2f}[/bold]\n[bold]Enter new temperature (0.1-2.0)[/bold]",
                                default=str(self.temperature)
                            )
                            new_temp = float(new_temp)
                            if 0.1 <= new_temp <= 2.0:
                                self.temperature = new_temp
                                self.tokens_data = self.token_analyzer.get_next_tokens(
                                    self.prompts[self.current_prompt_idx], 
                                    self.max_tokens, 
                                    self.temperature, 
                                    self.top_p
                                )
                                self.console.print(f"[green]Temperature set to: {self.temperature:.2f}[/green]")
                            else:
                                self.console.print("[red]Temperature must be between 0.1 and 2.0[/red]")
                        except ValueError:
                            self.console.print("[red]Invalid input. Temperature unchanged.[/red]")
                        live.start()
                    elif key in ('p', 'P'):  # Adjust top-p
                        live.stop()
                        try:
                            new_top_p = Prompt.ask(
                                f"[bold]Current top-p: {self.top_p:.2f}[/bold]\n[bold]Enter new top-p (0.01-1.0)[/bold]",
                                default=str(self.top_p)
                            )
                            new_top_p = float(new_top_p)
                            if 0.01 <= new_top_p <= 1.0:
                                self.top_p = new_top_p
                                self.tokens_data = self.token_analyzer.get_next_tokens(
                                    self.prompts[self.current_prompt_idx], 
                                    self.max_tokens, 
                                    self.temperature, 
                                    self.top_p
                                )
                                self.console.print(f"[green]Top-p set to: {self.top_p:.2f}[/green]")
                            else:
                                self.console.print("[red]Top-p must be between 0.01 and 1.0[/red]")
                        except ValueError:
                            self.console.print("[red]Invalid input. Top-p unchanged.[/red]")
                        live.start()
                    elif key == 'up':
                        if self.tokens_data:
                            self.selected_token_idx = (self.selected_token_idx - 1) % len(self.tokens_data)
                    elif key == 'down':
                        if self.tokens_data:
                            self.selected_token_idx = (self.selected_token_idx + 1) % len(self.tokens_data)
                    elif key == 'b':  # New: Token Influence Analysis
                        self.analyze_token_influence(live)
                    elif key == 'g':  # New: Start Token Game
                        self.start_token_game(live)
                    elif self.game_active and key in ('\r', '\n'):  # Game turn
                        self.play_game_turn(live)

                    live.update(self.ui_handler.build_interface(
                        self.prompts,
                        self.current_prompt_idx,
                        self.tokens_data,
                        self.selected_token_idx,
                        self.max_tokens,
                        self.search_query,
                        self.filters,
                        self.token_analyzer,
                        self.temperature,
                        self.top_p
                    ))

                except KeyboardInterrupt:
                    self.console.print("\n[bold green]Goodbye![/bold green]")
                    break
                except Exception as e:
                    self.console.print(f"[red]Error: {str(e)}[/red]")