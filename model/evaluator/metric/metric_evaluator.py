import csv
import os
import ast
import re
from dataloader import load_dataset_from_file
from scipy.stats import spearmanr, pearsonr
import numpy as np
import pandas as pd
import math

# ## final
## change path and folder names
csv_filepath = f'../output_csv_results/pvq21/llama3-8b-sft-full-data-prompt1-pvq21-1epoch.csv'
eval_dataset = load_dataset_from_file(f'../../evaluator/pvq21/test_data_prompt1.json')


# Load model output
def load_model_output(csv_filepath, generated_prompts):
    with open(csv_filepath, mode='r') as file:
        reader = csv.reader(file)
        for row in reader:
            row_content = row[0]
            dict_list = ast.literal_eval(row_content)
            if isinstance(dict_list, list) and dict_list:
                generated_text = dict_list[0].get('generated_text', '')
                generated_prompts.append(generated_text)


# Collect GT and predicted scores
def collect_result(generated_prompts, eval_dataset, total_gt_list, total_pred_list):
    current_input = None
    current_gt_list = []
    current_pred_list = []
    missmatch = 0

    for i in range(len(generated_prompts)):
        gt_score = eval_dataset[i]['output']
        generated_text = generated_prompts[i]
        response = generated_text.split("\n### Response:")[1].strip()
        match = re.search(r"\b\d+(\.\d{1,3})?\b", response)
        if match:
            predicted_score = float(match.group())
        else:
            missmatch += 1
            continue
        
        # 4≤ score ≤ 6 → 0, 1, 2, 3/7, 8, 9, 10
        # 3≤ score ≤7  → 0, 1, 2/8, 9, 10
        # 3≤ score ≤6  → 0, 1, 2,/7, 8, 9, 10
        # 4≤ score ≤ 7 → 0,1,2,3/8,9,10
        
        # remove 3 ≤ gt_score ≤ 7
        # if 3 <= gt_score <= 7:
        #     continue

        input_value = eval_dataset[i]['input']['message']

        if input_value != current_input:
            if current_gt_list:
                total_gt_list.append(current_gt_list)
            if current_pred_list:
                total_pred_list.append(current_pred_list)
            current_gt_list = []
            current_pred_list = []
            current_input = input_value

        current_gt_list.append(gt_score)
        current_pred_list.append(predicted_score)
    
    if current_gt_list:
        total_gt_list.append(current_gt_list)
    if current_pred_list:
        total_pred_list.append(current_pred_list)

    print("missmatch: ", missmatch)


# RMSE calculation
def RMSE(total_gt_list, total_pred_list):
    flattened_gt = [item for sublist in total_gt_list for item in sublist]
    flattened_pred = [item for sublist in total_pred_list for item in sublist]

    rmse_overall = sum((g - p) ** 2 for g, p in zip(flattened_gt, flattened_pred))
    return math.sqrt(rmse_overall / len(flattened_gt))

# Accuracy calculation
def Accuracy(total_gt_list, total_pred_list):
    flattened_gt = [item for sublist in total_gt_list for item in sublist]
    flattened_pred = [item for sublist in total_pred_list for item in sublist]

    accuracy = sum(1 for g, p in zip(flattened_gt, flattened_pred) if g == p)
    return accuracy / len(flattened_gt)

# Spearman correlation calculation
def SpearmanCorr(total_gt_list, total_pred_list):
    flattened_gt = [item for sublist in total_gt_list for item in sublist]
    flattened_pred = [item for sublist in total_pred_list for item in sublist]

    correlation, _ = spearmanr(flattened_gt, flattened_pred)
    return correlation

# Pearson correlation calculation
def PearsonCorr(total_gt_list, total_pred_list):
    flattened_gt = [item for sublist in total_gt_list for item in sublist]
    flattened_pred = [item for sublist in total_pred_list for item in sublist]

    correlation, _ = pearsonr(flattened_gt, flattened_pred)
    return correlation

# NDCG calculation
def dcg(scores):
    return np.sum([
        (2**rel - 1) / np.log2(idx + 2)
        for idx, rel in enumerate(scores)
    ])

def NDCG(total_gt_list, total_pred_list):
    ndcg_scores = []
    for i in range(len(total_gt_list)):
        top_10_indices = sorted(
            range(len(total_pred_list[i])),
            key=lambda j: total_pred_list[i][j],
            reverse=True
        )[:10]

        top_10_gt_values = [total_gt_list[i][j] for j in top_10_indices]
        top_10_ideal_values = sorted(total_gt_list[i], reverse=True)[:10]

        idcg = dcg(top_10_ideal_values)
        actual_dcg = dcg(top_10_gt_values)
        
        if idcg == 0:
            ndcg = 0
        else:
            ndcg = actual_dcg / idcg
        ndcg_scores.append(ndcg)

    average_ndcg = np.mean(ndcg_scores)
    print("Average NDCG for messages:", average_ndcg)

    flattened_gt = [item for sublist in total_gt_list for item in sublist]
    flattened_pred = [item for sublist in total_pred_list for item in sublist]

    top_20_indices = sorted(
        range(len(flattened_pred)),
        key=lambda j: flattened_pred[j],
        reverse=True
    )[:20]
    top_20_gt_values = [flattened_gt[j] for j in top_20_indices]
    top_20_ideal_values = sorted(flattened_gt, reverse=True)[:20]

    idcg = dcg(top_20_ideal_values)
    actual_dcg = dcg(top_20_gt_values)
    if idcg == 0:
        overall_ndcg = 0
    else:
        overall_ndcg = actual_dcg / idcg

    print("Average NDCG:", average_ndcg)
    return average_ndcg

# Calculate overall scores
def calculate_overall_scores(total_gt_list, total_pred_list):
    rmse = RMSE(total_gt_list, total_pred_list)
    accuracy = Accuracy(total_gt_list, total_pred_list)
    spearman = SpearmanCorr(total_gt_list, total_pred_list)
    pearson = PearsonCorr(total_gt_list, total_pred_list)
    ndcg = NDCG(total_gt_list, total_pred_list)

    return rmse, accuracy, spearman, pearson, ndcg

# Main execution
generated_prompts = []
load_model_output(csv_filepath, generated_prompts)

total_gt_list = []
total_pred_list = []
collect_result(generated_prompts, eval_dataset, total_gt_list, total_pred_list)

# Overall score calculation
rmse, accuracy, spearman, pearson, ndcg = calculate_overall_scores(total_gt_list, total_pred_list)

# Print overall scores
print("Overall Scores for the Entire Dataset:")
print(f"RMSE: {rmse:.4f}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Spearman Correlation: {spearman:.4f}")
print(f"Pearson Correlation: {pearson:.4f}")
print(f"NDCG: {ndcg:.4f}")
