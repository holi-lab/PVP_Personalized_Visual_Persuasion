# DALL-E Image Generator Script

This Python script demonstrates how to generate images using OpenAI's DALL-E 3 model, based on a given premise and message. It uses GPT-4 to create a detailed image description, which is then used as input for DALL-E 3.

## Features

- Generate image descriptions using GPT-4
- Create images using DALL-E 3
- Save generated images locally
- Retry mechanism for API calls

## Prerequisites

- Python 3.x
- OpenAI Python library
- Requests library

## Installation

1. Ensure you have Python 3.x installed on your system.
2. Install the required libraries:

   ```
   pip install openai requests
   ```

3. Set up your OpenAI API key as an environment variable:

   ```
   export OPENAI_API_KEY='your-api-key-here'
   ```

## Usage

1. Modify the `message` and `premise` variables in the script to your desired values.

2. Run the script:

   ```
   python dalle_generate_image.py
   ```

3. The script will:
   - Generate an image description based on your premise and message
   - Use DALL-E 3 to create an image
   - Save the image as `generated_image.jpg` in the current directory

## Functions

- `dalle_call(prompt)`: Calls DALL-E 3 API to generate an image
- `save_image(url)`: Downloads and saves the generated image
- `make_query(premise, message)`: Uses GPT-4 to create a detailed image description
- `generate_image(premise, message)`: Orchestrates the image generation process

## Customization

- You can modify the `size` parameter in the `dalle_call` function to change the image dimensions.
- Adjust the retry mechanism by changing the `max_retries` value in various functions.

## Note

- This script requires an OpenAI API key with access to both GPT-4 and DALL-E 3 models.
- API usage may incur costs. Please check OpenAI's pricing page for current rates.

## Error Handling

The script includes basic error handling and will retry API calls up to 3 times before raising an error.
