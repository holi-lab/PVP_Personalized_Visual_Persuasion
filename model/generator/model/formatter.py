import json


def read_json_file(file_path):
    """Reads a JSON file and returns the data."""
    with open(file_path, "r") as file:
        return json.load(file)


def format_data(obj):
    """Formats a single JSON object into the specified string format."""
    instruction = obj.get("instruction", "")
    message = obj.get("input", {}).get("message", "")
    value = obj.get("input", {}).get("value", "")

    formatted_string = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
Message: {message}
Value: {value}

Just directly output the image description without adding any prefixes or other modifiers.
"""

    return formatted_string


def format_json_file(file_path):
    """Reads a JSON file and formats each object, returning a list of formatted strings."""
    data = read_json_file(file_path)
    return [format_data(obj) for obj in data]
