# Token Explorer Architecture

This document explains the architecture and design of the Token Explorer application.

## Overview

Token Explorer is a command-line interface (CLI) application that allows users to explore how large language models (LLMs) predict the next token from a given prompt. The application displays the probability distribution of the next possible tokens and provides an interactive environment to experiment with different prompts and models.

## Architecture

The application is organized into several modules, each responsible for a specific aspect of the functionality:

### 1. Main Application (`token_explorer.py`)

The main application class orchestrates the functionality of the entire application. It manages:
- Application state (current prompt, selected token, etc.)
- Model loading and switching
- Token analysis
- UI updates
- Keyboard input processing

### 2. Configuration (`config.py`)

Handles configuration settings through the `TokenExplorerConfig` dataclass, including:
- Default model name
- Maximum number of tokens to show
- Default prompts
- UI theme
- Sampling parameters

### 3. Model Management (`model_manager.py`)

Manages model loading and provides an abstraction over different model providers through the `ModelManager` class. This allows for potential support of multiple providers (Hugging Face, OpenAI, etc.) in the future.

### 4. Model Provider (`provider.py`)

Implements the `BaseProvider` abstract class with a specific implementation for Hugging Face models (`HuggingFaceProvider`). This class handles:
- Loading models and tokenizers
- Getting next token predictions
- Getting model metadata
- Managing model state

### 5. Token Analysis (`token_analyzer.py`)

Performs the analysis of token probabilities and provides statistical insights. The `TokenAnalyzer` class includes functionality for:
- Getting next token predictions with caching
- Calculating entropy and perplexity
- Calculating token statistics
- Analyzing token influence
- Generating probability visualizations

### 6. UI Handling (`ui_handler.py`)

Handles all aspects of the user interface using the `rich` library. The `UIHandler` class manages:
- Building the terminal UI layout
- Rendering token tables
- Displaying prompts and controls
- Managing themes
- Creating panels for different UI elements

### 7. Input Handling (`input_handler.py`)

Provides cross-platform keyboard input handling through the `InputHandler` class. This ensures the application works on both Windows and Unix-like systems.

### 8. Caching (`cache.py`)

Implements a simple LRU (Least Recently Used) cache using the `TokenCache` class to store tokenization results and improve performance.

### 9. Entry Point (`main.py`)

The entry point of the application that initializes and runs the TokenExplorer.

## Key Features

### Interactive Token Exploration
Users can see the next most probable tokens for a prompt along with their probabilities and select one to append to the current prompt.

### Model Switching
The application supports switching between different Hugging Face models to compare token predictions.

### Token Analysis
The application provides entropy, perplexity, and other statistical metrics for the token distribution.

### Sampling Parameters
Users can adjust temperature and top-p (nucleus) sampling parameters to see how they affect token probabilities.

### Theme Support
The UI supports multiple themes (default, light, dark) for better visual experience.

### Game Mode
An interactive game mode allows users to construct a target prompt by selecting tokens strategically.

## Design Principles

1. **Modularity**: Each file has a clear responsibility, making the code easier to understand and maintain.

2. **Extensibility**: The provider abstraction allows for adding support for other model APIs in the future.

3. **Performance**: Token caching improves responsiveness when revisiting similar prompts.

4. **User Experience**: Rich terminal UI with keyboard navigation makes the exploration intuitive.

## Installation and Usage

See the main README.md file for installation and usage instructions.