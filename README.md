# Token Explorer

A simple CLI app that lets you see how LLMs pick the next token and the token probability distribution, pick the next token yourself, add custom prompts and more.

## Features

- **Interactive Token Exploration**: Navigate through the next most probable tokens for any given prompt
- **Model Switching**: Load and switch between different Hugging Face models
- **Prompt Management**: Add, save, and navigate between multiple prompts
- **Token Information**: Inspect detailed information about specific tokens (ID, bytes, encoding, etc.)
- **Model Metadata**: View detailed information about the loaded model
- **Search & Filter**: Search and filter tokens by content or type (word, punctuation, number)
- **Entropy Analysis**: View entropy, perplexity, and other statistical metrics of token distributions
- **Theme Support**: Toggle between different UI themes (default, light, dark)
- **Sampling Parameters**: Adjust temperature and top-p (nucleus) sampling parameters
- **Token Influence Analysis**: Analyze how adding specific tokens affects the next predictions
- **Token Exploration Game**: Interactive game to construct specific target prompts
- **History & Undo**: Undo previous token additions and view prompt history

## Installation

1. Clone or download this repository to your local machine.
2. Install the required dependencies using pip:

```bash
pip install torch transformers rich
```

3. Run the application:

```bash
python main.py
```

## Requirements

- Python 3.7+
- torch
- transformers
- rich

## Usage

Once the application is running, you'll see the main interface with:
- The current prompt at the top
- A table showing the next most probable tokens
- Controls panel on the right
- Entropy and statistical information at the bottom

Use the following keyboard controls:

| Key | Action |
| --- | --- |
| `↑`/`↓` | Navigate tokens |
| `←`/`→` or `n`/`p` | Previous/Next prompt |
| `Enter` | Append selected token to prompt |
| `u` | Undo last token addition |
| `a` | Add a new prompt |
| `m` | Switch model |
| `i` | Inspect selected token details |
| `h` | View prompt history |
| `c` | View model metadata |
| `s` | Save current session to file |
| `f` | Filter tokens by type (word, punctuation, number) |
| `/` | Search tokens by content |
| `e` | View entropy analysis |
| `t` | Toggle UI theme (default/light/dark) |
| `V` | Adjust temperature sampling parameter |
| `P` | Adjust top-p (nucleus) sampling parameter |
| `b` | Analyze token influence |
| `g` | Start token exploration game |
| `q` | Quit the application |

## Project Structure

```
Token_explorer/
├── cache.py              # Token caching mechanism
├── config.py             # Configuration management
├── input_handler.py      # Keyboard input handling
├── main.py               # Entry point of the application
├── model_manager.py      # Model loading and management
├── provider.py           # Abstract provider interface and Hugging Face implementation
├── token_analyzer.py     # Token analysis and prediction logic
├── token_explorer.py     # Main application logic
├── ui_handler.py         # UI rendering and display
├── README.md             # This file
└── QWEN.md               # Project instructions
```

## Configuration

The application uses a `TokenExplorerConfig` dataclass to manage settings such as:
- Default model name
- Maximum number of tokens to show
- Default prompts
- UI theme
- Sampling parameters (temperature, top-p)

## Architecture

The application follows a modular architecture:

- **Provider System**: Abstracted interface for different AI model providers (currently implemented for Hugging Face)
- **Model Manager**: Handles model loading and metadata retrieval
- **Token Analyzer**: Processes token probabilities and provides statistical analysis
- **UI Handler**: Manages the rich terminal interface
- **Input Handler**: Cross-platform keyboard input handling
- **Cache System**: Caches tokenization results for performance


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with [Hugging Face Transformers](https://github.com/huggingface/transformers)
- UI rendered with [Rich](https://github.com/Textualize/rich)
- Powered by [PyTorch](https://pytorch.org/)