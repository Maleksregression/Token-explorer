# Token Explorer Usage Examples

This document provides examples of how to use the Token Explorer application effectively.

## Basic Navigation

1. **Starting the Application**:
   ```bash
   python main.py
   ```

2. **Navigating Between Tokens**:
   - Use the UP (↑) and DOWN (↓) arrow keys to select different tokens in the probability list
   - The selected token will be highlighted

3. **Adding Tokens to Prompt**:
   - Press ENTER to append the selected token to your current prompt
   - The token table will update with new predictions based on the modified prompt

4. **Switching Between Prompts**:
   - Use LEFT (←) and RIGHT (→) arrow keys, or 'p' and 'n' keys to navigate between saved prompts
   - Add new prompts with the 'a' key

## Model Management

1. **Switching Models**:
   - Press 'm' to open the model selection interface
   - Choose from the available models or enter a Hugging Face model name directly
   - New models will be loaded and token predictions will update

2. **Viewing Model Information**:
   - Press 'c' to view detailed metadata about the currently loaded model

## Advanced Features

1. **Filtering Tokens**:
   - Press 'f' to filter tokens by type (word, punctuation, number)
   - This helps focus on specific kinds of tokens in the prediction list

2. **Searching Tokens**:
   - Press '/' to search for specific tokens by content
   - This filters the token list to show only those matching your search term

3. **Entropy Analysis**:
   - Press 'e' to view detailed entropy, perplexity, and other statistical metrics
   - This gives insight into the confidence and diversity of the model's predictions

4. **Adjusting Sampling Parameters**:
   - Press 'V' to adjust the temperature parameter (controls randomness)
   - Press 'P' to adjust the top-p (nucleus) sampling parameter (controls diversity)
   - Changes will immediately affect the token probabilities

5. **Token Influence Analysis**:
   - Press 'b' to analyze how adding specific tokens affects the next predictions
   - This shows how each token choice influences the model's next predictions

6. **Theme Switching**:
   - Press 't' to cycle between UI themes (default, light, dark)
   - This changes the color scheme of the interface

## Game Mode

1. **Starting the Game**:
   - Press 'g' to start the token exploration game
   - Enter a target prompt you want to construct

2. **Playing the Game**:
   - Select tokens that you think will get you closer to the target prompt
   - Each turn shows your current score based on similarity to the target
   - You have a limited number of turns to get as close as possible

## Saving and History

1. **Viewing History**:
   - Press 'h' to see the history of your prompt modifications

2. **Undoing Changes**:
   - Press 'u' to undo the last token addition to the current prompt

3. **Saving Sessions**:
   - Press 's' to save your current prompts to a file

## Tips for Better Exploration

1. **Starting Prompts**:
   - Begin with specific or contextual prompts for more interesting predictions
   - Try different prompt styles (questions, statements, incomplete sentences)

2. **Token Selection Strategy**:
   - Observe how different token choices lead to different directions in the text
   - High probability tokens tend to be safer/expected choices
   - Lower probability tokens can lead to more creative or unexpected results

3. **Sampling Parameters**:
   - Lower temperatures (0.1-0.7) make the model more confident and predictable
   - Higher temperatures (0.8-2.0) make the model more creative and random
   - Lower top-p values (0.1-0.5) focus on high-probability options
   - Higher top-p values (0.8-1.0) include more diverse options

4. **Analysis**:
   - Use entropy to understand how confident the model is in its predictions
   - High entropy suggests more uncertainty and diverse options
   - Low entropy suggests more certainty with one clear option