# Evaluator and Generator Scripts

This repository provides Python scripts for processing datasets, training models, and evaluating results for both the Evaluator and Generator. The scripts support dataset preprocessing, model fine-tuning, and performance evaluation.

## Features

- Preprocessing datasets for training the Evaluator or Generator
- Fine-tuning the Evaluator or Generator
- Running inference and evaluating metrics

## Prerequisites

- Python 3.12
- Hugging Face library
- Requests library in requirements.txt

## Installation

1. Ensure you have Python 3.12 installed on your system.
2. Install the required libraries:

   ```
   pip install -r requirements.txt
   ```

## Usage

1. Dataset preprocessing
- The datasets used in our paper are provided in `./data_preprocessing/data`.  
- Convert CSV files to JSON using the provided scripts:  
  - `prompt_csv_to_json.py` or `prompt_generator.py`.  
  - This process prepares the dataset for both the evaluator and generator.  
- If you wish to modify the prompts, you can edit them in `prompt_csv_to_json.py` or `prompt_generator.py`.


2. Preparing the Evaluator(Generator)
- In the `evaluator/`(`generator/`) folder, you will find the `data/` subfolder.  
  - The generated JSON files must be placed in this folder.  
  - The folder currently contains the JSON datasets used in our paper. 

3. Configuration Setup.
- Input your access key for Hugging Face model at *.yaml at `config/` folder.

4. Running the Fine-Tuning Script
Run the script using provided command. The bash file name is just an example.

   ```
   bash run_training_pvp21.sh  
   ```

5. The script will:
   - Fine-tuning Evaluator(Generator) using train dataset.
   - Running Inference the results.
      - Result of the Evaluator's Inference: Scores
      - Result of the Generator's Inference: Descriptions

4. Evaluating Model Performance
To evaluate the model’s performance, run the provided metric script:
   ```
      python ./metric/metric_evaluator.py
   ```
This script calculates the following key evaluation metrics:
	•	Root Mean Square Error (RMSE)
	•	Accuracy
	•	Spearman and Pearson Correlations
	•	Normalized Discounted Cumulative Gain (NDCG)

Note: Ensure that the generator results have been inferred using the evaluator before running the evaluation script.


## Customization

- The YAML configuration files in the config/ folder define the fine-tuning and inference settings for the LLaMA 3 8B model, including:
- Hyperparameters (e.g., learning rate, batch size, epochs)
- Logging configurations (e.g., WandB integration)
- Quantization and PEFT (LoRA) settings

You can customize these settings according to your specific requirements.

