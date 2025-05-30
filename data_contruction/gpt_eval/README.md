# GPT Image Evaluation Script

This script uses OpenAI's GPT-4 to evaluate images based on their intended message and specific persuasion strategies.

## Features

- Image encoding for API consumption
- Two-turn GPT-4 conversation for image evaluation
- Multiple persuasion strategy support
- Evaluation score and reasoning output

## Prerequisites

- Python 3.6+
- OpenAI Python library
- OpenAI API key (fees may apply)

## Setup

1. Install the required package:
   ```
   pip install openai
   ```

2. Set your OpenAI API key:
   ```
   export OPENAI_API_KEY='your-api-key-here'
   ```

## Usage

Run the script with:
```
python gpteval_image.py
```

The script includes an example case:
- Message: "Do not smoke"
- Premise: "Do not smoke. Then, you enjoy the freshness in your breath and clothes."
- Strategy: xReact

Modify the `message`, `premise`, `strategy`, and `image_path` variables in the script for your use case.

## Persuasion Strategies

- `External Emotion`: Emotional responses that other people may experience.
- `Internal Emotion`: Emotional reactions the viewer may personally experience.
- `Bandwagon`: How popular the target behavior is among other people.
- `Consequence`: Consequences other than perceived persona and emotional responses (e.g., harms, wealth).
- `Perceived Persona`: How the viewer’s persona or attributes would be perceived by others.

## Note
Ensure compliance with OpenAI's use-case policies and sufficient API credits.
