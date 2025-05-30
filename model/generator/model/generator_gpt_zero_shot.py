from openai import OpenAI
from formatter import format_json_file
import json
from tqdm import tqdm  # tqdm 라이브러리 임포트
import os
from dotenv import load_dotenv
import argparse

load_dotenv()

# Set up your API key
api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)

# parser = argparse.ArgumentParser(description='Load model name')
# parser.add_argument('--model', type=str, required=True, help='Model name')
# args = parser.parse_args()

model_name = "gpt-4o"

# Function to perform inference on a batch of prompts using ChatCompletion
def gpt_inference(prompts):
    responses = []
    system_message = "You are an helpful AI assistant for generating image description."
    for prompt in tqdm(prompts, desc="Processing prompts"):  # tqdm 적용
        response = client.chat.completions.create(
            model=model_name,  # Use the GPT-4o mini model
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt},
            ],
            max_tokens=300,
            n=1,
            stop=None,
            temperature=1.0,
        )
        responses.append(response.choices[0].message.content.strip())
    return responses


def save_responses_to_json(formatted_data, responses, output_path):
    """Saves the formatted prompts and their responses to a specified JSON output file."""
    combined_data = [
        {"prompt": formatted_data[i], "generated_text": responses[i]}
        for i in range(len(formatted_data))
    ]
    with open(output_path, "w") as file:
        json.dump(combined_data, file, indent=4)


def main():
    # file_path = "./data/test_data.json"
    file_path_prompt2 = "./data/generator/generator_test.json"
    output_path_prompt2 = f"./output_csv_results/gpt.json"

    file_path_list = [file_path_prompt2]
    output_path_list = [output_path_prompt2]

    for i in range(len(file_path_list)):
        file_path = file_path_list[i]
        output_path = output_path_list[i]

        # JSON 파일 포맷팅
        formatted_data = format_json_file(file_path)

        # Perform batch inference
        batch_responses = gpt_inference(formatted_data)

        # Save responses to a JSON file
        save_responses_to_json(formatted_data, batch_responses, output_path)
        print(f"Responses have been saved to {output_path}")


if __name__ == "__main__":
    main()
