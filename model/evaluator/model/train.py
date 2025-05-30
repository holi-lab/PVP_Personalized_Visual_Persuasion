import setproctitle
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from dataloader import load_dataset_from_file, reformat_dataset
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
import huggingface_hub
import wandb
import yaml
import argparse
from transformers import TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig

# Ensure deterministic behavior for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def main():

    # Argument parser to accept the config file path
    parser = argparse.ArgumentParser(description='Load experiment configuration.')
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration YAML file.')
    args = parser.parse_args()

    if torch.cuda.is_available():
        current_device = torch.cuda.current_device()
        print(f"Current CUDA device number: {current_device}")
    else:
        print("CUDA is not available. Using CPU.")

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
    dataset_type = config['data_process']['dataset_type']
    base_model = config['model']['base_model']
    response_template = config['data_process']['response_template']
    wandb_project = config['wandb']['project']
    wandb_entity = config['wandb']['entity']
    wandb_run_name = config['wandb']['run_name']
    max_seq_length = config['data_process']['max_seq_length']
    save_model_path = config['path']['save_model_path']

    # Check CUDA capability and set configurations accordingly
    if torch.cuda.get_device_capability()[0] >= 8:
        attn_implementation = "flash_attention_2"
        torch_dtype = torch.bfloat16
        print("Using flash_attention_2 implementation and bfloat16 dtype.")
    else:
        attn_implementation = "eager"
        torch_dtype = torch.float16
        print("Using eager implementation and float16 dtype.")

    # QLoRA configuration
    quant_config = BitsAndBytesConfig(
        load_in_4bit=config['quantization']['load_in_4bit'],
        bnb_4bit_quant_type=config['quantization']['bnb_4bit_quant_type'],
        bnb_4bit_compute_dtype=torch_dtype,
        bnb_4bit_use_double_quant=config['quantization']['bnb_4bit_use_double_quant'],
    )

    # PEFT (LoRA) configuration
    peft_params = LoraConfig(
        lora_alpha=config['peft']['lora_alpha'],
        lora_dropout=config['peft']['lora_dropout'],
        r=config['peft']['r'],
        bias=config['peft']['bias'],
        task_type=config['peft']['task_type'],
        target_modules=config['peft']['target_modules']
    )

    # Training arguments
    training_args = TrainingArguments(**config['training_args'])

    ################################
    # Set seed for reproducibility #
    ################################

    set_seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Set the process name for easier identification
    setproctitle.setproctitle(process_name)

    # Login to Hugging Face
    huggingface_hub.login(hf_token)

    ################
    # Load Dataset #
    ################

    train_dataset = load_dataset_from_file(train_dataset_path).shuffle(seed=seed)  # Explicit shuffle needed
    formatted_train_dataset = reformat_dataset(train_dataset, is_train=True, dataset_type=dataset_type)

    eval_dataset = load_dataset_from_file(eval_dataset_path).shuffle(seed=seed)  # Explicit shuffle needed
    formatted_eval_dataset = reformat_dataset(eval_dataset, is_train=True, dataset_type=dataset_type)

    # Print first item in each dataset
    print("First item in the formatted_train_dataset:", formatted_train_dataset[0])
    print("First item in the formatted_eval_dataset:", formatted_eval_dataset[0])

    ############################
    # Load model and tokenizer #
    ############################
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        attn_implementation=attn_implementation,
        quantization_config=quant_config,
        torch_dtype=torch_dtype,  # Explicitly set dtype
        use_cache=False,  # Disable use_cache
    )
    model.config.use_cache = False  # Also disable use_cache in the model config
    model.pretraining_tp = 1

    print(model.dtype)  # Print current dtype

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Prepare data collator
    data_collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    # Initialize Weights & Biases logging
    wandb.init(project=wandb_project, entity=wandb_entity, name=wandb_run_name)

    #########################
    # Initialize SFTTrainer #
    #########################
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=formatted_train_dataset,
        eval_dataset=formatted_eval_dataset,
        peft_config=peft_params,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        args=training_args,
        packing=False,
        data_collator=data_collator,
        dataset_kwargs={"add_special_tokens": False},
    )

    ############
    # Training #
    ############
    
    trainer.train()

    #################################################
    # Save PEFT Model, Full model and trainer state #
    #################################################

    trainer.model.save_pretrained(save_model_path)
    model.save_pretrained(save_model_path)
    trainer.save_state()
    tokenizer.save_pretrained(save_model_path)

    # Finish WandB run
    wandb.finish()

if __name__ == "__main__":
    main()
