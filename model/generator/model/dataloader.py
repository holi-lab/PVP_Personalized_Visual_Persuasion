from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
# import config
import huggingface_hub

# Initialize tokenizer
huggingface_hub.login("your_token")

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", trust_remote_code=True)

def load_dataset_from_file(dataset_path):
    return load_dataset('json', data_files=dataset_path, split="train") ## there is only one split, and its default setting is train. so we use train split for both train_dataset and eval_dataset. do not confuse

def formatting_prompts_func(examples, is_train, dataset_type):
    system_message = "You are an helpful AI assistant for generating image description."

    alpaca_prompts = {
        "value": """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
Message: {}
Value: {}

Just directly output the image description without adding any prefixes or other modifiers.
""",
        "big5": """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
Message: {}
Big5: {}

Just directly output the image description without adding any prefixes or other modifiers.
""",
        "mfq": """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
Message: {}
MFQ: {}

Just directly output the image description without adding any prefixes or other modifiers.
""",
        "none": """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
Message: {}

Just directly output the image description without adding any prefixes or other modifiers.
"""
    }
    
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]
    chat_template_texts = []
    
    for instruction, input, output in zip(instructions, inputs, outputs):
        # Extract fields from the input based on the dataset type
        message = input["message"]
        image_description = input["image_description"]

        if dataset_type == "value":
            value = input["value"]
            user_message = alpaca_prompts["value"].format(instruction, message, value)
        elif dataset_type == "big5":
            big5 = input["big5"]
            user_message = alpaca_prompts["big5"].format(instruction, message, big5)
        elif dataset_type == "mfq":
            mfq = input["mfq"]
            user_message = alpaca_prompts["mfq"].format(instruction, message, mfq)
        else:  # None
            user_message = alpaca_prompts["none"].format(instruction, message)

        if is_train:
            # Create the conversation structure in case of training
            conversation = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": "### Response:" + " " + image_description} ## 공백이 없으면 response key를 못찾음. 이게 연속된 단어로 인식되어서 tokenize되는 건가봄
            ]
        else:
            # Create the conversation structure in case of evaluation
            conversation = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ]

        # Apply chat template
        chat_template_text = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=not is_train)  # if is_train is True, add_generation_prompt is False and vice versa
        if not is_train:
            chat_template_text += "### Response:"
        
        chat_template_texts.append(chat_template_text)
    
    return {"text": chat_template_texts}


def reformat_dataset(dataset, is_train, dataset_type):
    return dataset.map(
        lambda examples: formatting_prompts_func(examples, is_train, dataset_type),
        batched=True,
        load_from_cache_file=False
    )