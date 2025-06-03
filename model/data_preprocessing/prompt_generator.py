import pandas as pd
import json

# Public path
num_threshold = 9
base_input_path = f"./data/"
base_output_path = f"./data/json_data/generator/over{num_threshold}"
input_file_name = "_paper"
output_file_name = "_paper"

# Data paths
data_paths = {
    "train": f"{base_input_path}/train{input_file_name}.csv",
    "eval": f"{base_input_path}/eval{input_file_name}.csv",
    "test": f"{base_input_path}/test{input_file_name}.csv",
    "generator_test": f"{base_input_path}/generator_test.csv"
}

# Output paths
output_paths = {
    "train": f"{base_output_path}/generator_train{output_file_name}_over{num_threshold}.json",
    "eval": f"{base_output_path}/generator_eval{output_file_name}_over{num_threshold}.json",
    "test": f"{base_output_path}/generator_test{output_file_name}_over{num_threshold}.json",
    "generator_test": f"{base_output_path}/generator_test.json"
}

# Score filtering (if needed)
def filter_by_score(df, num=None):
    if num is not None:
        return df[df['Score'] >= num]
    return df

# Function for processing and saving data
def process_data(input_path, output_path, num=None):
    # Read CSV file
    df = pd.read_csv(input_path, encoding='utf-8')

    # Filter by the given score threshold
    df = filter_by_score(df, num)

    # Create a list for JSON conversion
    result_list = []

    for index, row in df.iterrows():
        message = row['Message']
        image_description = row['query']
        value = row['pvq21']
        mfq = row['mfq']
        big5 = row['big5']
        persuasiveness_score = row['Score']

        dict_element = {
            "instruction": """Generate an image description based on the following task. You have received a message and an individual’s values as input. These values are based on Schwartz’s 10 basic values, rated from 1 to 6, with higher scores indicating greater importance to the individual. Craft an image description that conveys the message’s intent using only visual elements like colors, symbols, or scenarios that resonate with the individual’s values. Do not include any references to visible text, such as banners, signs, or posters with wording. The description should rely solely on non-verbal cues and should not exceed 10 sentences.""",
            "input": {
                "message": message,
                "value": eval(value) if isinstance(value, str) else value,
                "mfq": eval(mfq) if isinstance(mfq, str) else mfq,
                "big5": eval(big5) if isinstance(big5, str) else big5,
                "image_description": image_description
            },
            "output": persuasiveness_score
        }

        result_list.append(dict_element)

    # Convert the list to JSON format
    json_result = json.dumps(result_list, indent=4)

    # Save the result as a JSON file
    with open(output_path, 'w') as json_file:
        json_file.write(json_result)

    print(f"Processed and saved data for {output_path}")

# Set score threshold (use if necessary)
# num_threshold = 9  # Desired threshold value

# Process data
process_data(data_paths["train"], output_paths["train"], num=num_threshold)
process_data(data_paths["eval"], output_paths["eval"], num=num_threshold)
# process_data(data_paths["test"], output_paths["test"])  # Process without filtering
process_data(data_paths["generator_test"], output_paths["generator_test"])  # Process without filtering
