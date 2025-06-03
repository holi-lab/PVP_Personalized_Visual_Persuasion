import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed, pipeline
from dataloader import load_dataset_from_file, reformat_dataset
from tqdm import tqdm
import yaml
import argparse
import os
import csv
import setproctitle

# Argument parser to accept the config file path
parser = argparse.ArgumentParser(description='Load experiment configuration.')
parser.add_argument('--config', type=str, required=True, help='Path to the configuration YAML file.')
args = parser.parse_args()

################
# Load Configs #
################

# Load YAML config
with open(args.config, 'r') as file:
    config = yaml.safe_load(file)

seed = config['training_args']['seed']
process_name = config['process']['process_name']
hf_token = config['huggingface']['token']
train_dataset_path = config['path']['train_dataset_path']
eval_dataset_path = config['path']['eval_dataset_path']
test_dataset_path = config['path']['test_dataset_path']
dataset_type = config['data_process']['dataset_type']
max_seq_length = config['data_process']['max_seq_length']
save_model_path = config['path']['save_model_path']
output_csv_filename = config['path']['output_csv_filename']
output_csv_dir = config['path']['output_csv_dir']
inference_batch_size = config['inference']['batch_size']


    ################################
    # Set seed for reproducibility #
    ################################

set_seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Set the process name
setproctitle.setproctitle(process_name)

if torch.cuda.is_available():
    current_device = torch.cuda.current_device()
    print(f"Current CUDA device number: {current_device}")
    torch.set_default_device(current_device)
else:
    print("CUDA is not available. Using CPU.")

# Check CUDA capability and set configurations
if torch.cuda.get_device_capability()[0] >= 8:
    attn_implementation = "flash_attention_2"
    torch_dtype = torch.bfloat16
    print("Using flash_attention_2 implementation and bfloat16 dtype.")
else:
    attn_implementation = "eager"
    torch_dtype = torch.float16
    print("Using eager implementation and float16 dtype.")


tokenizer = AutoTokenizer.from_pretrained(save_model_path)
model = AutoModelForCausalLM.from_pretrained(save_model_path, attn_implementation=attn_implementation, torch_dtype=torch_dtype)

if hasattr(model.config, "quantization_config"):
    print("Model is quantized.")
else:
    print("Model is not quantized.")

device = next(model.parameters()).device
print(f"The model is on device: {device}")


print(model.config)
print(model)

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

# Load and reformat the dataset
test_dataset = load_dataset_from_file(test_dataset_path)
formatted_test_dataset = reformat_dataset(test_dataset, is_train=False, dataset_type=dataset_type)


batch_size = inference_batch_size

# Initialize the text generation pipeline
pipe = pipeline(
    task="text-generation", model=model, tokenizer=tokenizer, max_length=max_seq_length, truncation=True, torch_dtype=torch_dtype, device=current_device
)

##################
# Inference Loop #
##################
# Generate responses with retry on CUDA OOM

batch_size = inference_batch_size

while True:
    try:
        total_batches = len(formatted_test_dataset) // batch_size
        if len(formatted_test_dataset) % batch_size != 0:
            total_batches += 1
        results = []
        for i in tqdm(range(total_batches), desc="Evaluation Progress"):
            batch_start = i * batch_size
            batch_end = min(batch_start + batch_size, len(formatted_test_dataset))
            batch_prompts = formatted_test_dataset['text'][batch_start:batch_end]
            if not batch_prompts:
                print("Error: batch_prompts is empty")
            # print(f"batch_prompts: {batch_prompts}")
            batch_results = pipe(batch_prompts, batch_size=batch_size, do_sample=False)
            results.extend(batch_results)
        break  # If successful, break out of the retry loop
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            # Reduce the batch size if OOM error occurs
            batch_size = batch_size // 2
            print(f"CUDA out of memory. Reducing batch size to {batch_size}.")
            if batch_size == 0:
                raise RuntimeError("Batch size has been reduced to 0. Cannot continue.")
        else:
            raise e
        
directory = output_csv_dir

if not os.path.exists(directory):
    os.makedirs(directory)

csv_filepath = os.path.join(directory, output_csv_filename)
with open(csv_filepath, mode = 'w', newline='') as file:
    writer = csv.writer(file)
    for i in range(len(results)):
        writer.writerow([results[i]])