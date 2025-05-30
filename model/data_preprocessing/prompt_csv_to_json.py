import pandas as pd
import json

# Set common paths
base_input_path = f"./data/"
base_output_path_template = f"./data/json_data/evaluator/{{condition}}"
input_file_name = f"_paper"
output_file_name_template = f"_data_prompt1_{{condition}}"

# Prompt user for condition input
condition = input("Enter condition (mfq30, pvq21, none, big5): ").strip().lower()

# List of valid conditions
valid_conditions = ["mfq30", "pvq21", "none", "big5"]

# Validate input condition
if condition not in valid_conditions:
    raise ValueError(f"Please enter a valid condition: {', '.join(valid_conditions)}")

# Set data paths
base_output_path = base_output_path_template.format(condition=condition)
output_file_name = output_file_name_template.format(condition=condition)

data_paths = {
    "train": f"{base_input_path}/train{input_file_name}.csv",
    "eval": f"{base_input_path}/eval{input_file_name}.csv",
    "test": f"{base_input_path}/test{input_file_name}.csv",
}

output_paths = {
    "train": f"{base_output_path}/train{output_file_name}.json",
    "eval": f"{base_output_path}/eval{output_file_name}.json",
    "test": f"{base_output_path}/test{output_file_name}.json",
}

# Instructions based on the selected condition
instructions = {
    "mfq30": """You will perform a task where you predict how persuasive certain individuals will find an image created from a message, rating it from 0 to 10. Please predict the persuasiveness score based on the image description and the user’s Moral Foundation scores. These scores are based on the Moral Foundations Questionnaire 30 (MFQ30), and each score reflects the individual’s importance placed on each moral foundation domain. Respond with a single number between 0 and 10.""",
    
    "pvq21": """You will perform a task where you predict how persuasive certain individuals will find an image created from a message, rating it from 0 to 10. Please predict the persuasiveness score based on the image description and the user’s values. These values are based on Schwartz’s 10 basic values, where each value is rated on a scale from 1 to 6. The higher the value, the more emphasis is placed on that value. Respond with a single number between 0 and 10.""",
    
    "none": """You will perform a task where you predict how persuasive certain individuals will find an image created from a message, rating it from 0 to 10. Please predict the persuasiveness score based on the image description. Respond with a single number between 0 and 10.""",
    
    "big5": """You will perform a task where you predict how persuasive certain individuals will find an image created from a message, rating it from 0 to 10. Please predict the persuasiveness score based on the image description and the individual’s Big 5 personal traits, where higher scores reflect stronger manifestations of the associated behaviors and emotions, with each trait being scored between 2 and 10. Respond with a single number between 0 and 10."""
}

# Function to process data
def process_data(input_path, output_path, condition):
    # Read the CSV file
    df = pd.read_csv(input_path, encoding='utf-8')

    # Create a list to store results
    result_list = []

    # Iterate through each row in the CSV file and add to the list
    for _, row in df.iterrows():
        message = row['Message']
        image_description = row['query']
        value = row['pvq21']
        mfq = row['mfq']
        big5 = row['big5']
        persuasiveness_score = row['Score']

        # Prepare input data
        input_data = {
            "message": message,
            "image_description": image_description
        }

        # Include condition-specific data
        if condition in ["mfq30", "big5", "pvq21"]:
            input_data["value"] = eval(value) if isinstance(value, str) else value
            input_data["mfq"] = eval(mfq) if isinstance(mfq, str) else mfq
            input_data["big5"] = eval(big5) if isinstance(big5, str) else big5

        dict_element = {
            "instruction": instructions[condition],
            "input": input_data,
            "output": persuasiveness_score
        }

        result_list.append(dict_element)

    # Convert list to JSON format
    json_result = json.dumps(result_list, indent=4)

    # Save the result as a JSON file
    with open(output_path, 'w', encoding='utf-8') as json_file:
        json_file.write(json_result)

    # Print completion message
    print(f"Processed and saved data for {output_path}")

# Execute data processing
for key in data_paths:
    process_data(data_paths[key], output_paths[key], condition)