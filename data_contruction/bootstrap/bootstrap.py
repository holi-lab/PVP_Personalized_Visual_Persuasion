from sklearn.utils import resample
from scipy.stats import spearmanr
import numpy as np
import pandas as pd
import openpyxl

class StatBootstrap:
    def __init__(self, data_original_path, data_comparison_path):
        self.data_original_path = data_original_path
        self.data_comparison_path = data_comparison_path
        self.data_dict, self.comparison = self.load_data()
        self.sample = None
        self.aggregation_result = None

    def load_data(self):
        data_original = pd.read_csv(self.data_original_path)
        data_comparison = pd.read_csv(self.data_comparison_path)
        # data_comparison = pd.read_excel(self.data_comparison_path)

        number_of_datasets = len(data_original.columns)

        data_dict = {
            f"data{i}": data_original[f"trial{i}"].tolist() 
            for i in range(1, number_of_datasets + 1) 
            if f"trial{i}" in data_original.columns
        }
        
        # Load comparison scores (e.g., human ratings)
        comparison = data_comparison["dalle"].tolist()
        return data_dict, comparison  # Return GPT-generated scores and comparison scores

    def bootstrap(self, sample_size=2):
        data_list = list(self.data_dict.values())
        self.sample_size = sample_size
        self.sample = resample(data_list, n_samples=self.sample_size)  # Sample `sample_size` datasets
        return self.sample

    def aggregation_bootstrap(self, method="Mean"):
        if method == "Mean":
            self.aggregation_result = [np.mean(values) for values in zip(*self.sample)]
        elif method == "Median":
            self.aggregation_result = [np.median(values) for values in zip(*self.sample)]
        elif method == "Max":
            self.aggregation_result = [np.max(values) for values in zip(*self.sample)]
        elif method == "Min":
            self.aggregation_result = [np.min(values) for values in zip(*self.sample)]
        return self.aggregation_result

    def spearman(self, return_p_value=False):
        correlation, p_value = spearmanr(self.aggregation_result, self.comparison)
        if return_p_value:
            return correlation, p_value
        else:
            return correlation

    def calculate_confidence_interval(self, trials, sample_size=2, lower_pct=2.5, upper_pct=97.5):
        correlations = []
        p_values = []

        for _ in range(trials):
            self.bootstrap(sample_size=sample_size)
            self.aggregation_bootstrap(method="Mean")
            correlation, p_value = self.spearman(return_p_value=True)
            correlations.append(correlation)
            p_values.append(p_value)

        correlations_sorted = np.sort(correlations)
        p_values_sorted = np.array(p_values)[np.argsort(correlations)]

        lower_bound_index = int(trials * lower_pct / 100)
        upper_bound_index = int(trials * upper_pct / 100)

        lower_bound = correlations_sorted[lower_bound_index]
        upper_bound = correlations_sorted[upper_bound_index]

        # Option 1: Average after slicing (commented out)
        # avg_correlation = np.mean(correlations_sorted[lower_bound_index:upper_bound_index+1])
        # avg_p_value_within_bounds = np.mean(p_values_sorted[lower_bound_index:upper_bound_index+1])

        # Option 2: Average over full results
        avg_correlation = np.mean(correlations_sorted)
        avg_p_value_within_bounds = np.mean(p_values_sorted)

        return lower_bound, upper_bound, avg_correlation, avg_p_value_within_bounds

def main(data_original_path, data_comparison_path):

    # Create object — data is loaded on initialization
    stat = StatBootstrap(data_original_path, data_comparison_path)
    number_of_datasets = len(stat.data_dict)

    results = []

    for sample_size in range(1, number_of_datasets + 1):
        lower_bound, upper_bound, avg_correlation, avg_p_value = stat.calculate_confidence_interval(
            trials=1000,
            sample_size=sample_size,
            lower_pct=2.5,
            upper_pct=97.5
        )
        confidence_interval_width = upper_bound - lower_bound
        results.append((
            sample_size,
            lower_bound,
            upper_bound,
            confidence_interval_width,
            avg_correlation,
            avg_p_value
        ))

    results_df = pd.DataFrame(results, columns=[
        'Sample size',
        'Lower Bound',
        'Upper Bound',
        'CI Width',
        'Spearman Correlation',
        'Spearman P-Value'
    ])
    results_df.to_csv('../../data/output/dalle/confidence_intervals_95.csv', index=False)
    print("Results saved to 'confidence_intervals_95.csv'")

if __name__ == "__main__":
    data_original_path = "../../data/input/dalle/summary_dalle.csv"
    data_comparison_path = "../../data/input/dalle/review_50.csv"
    main(data_original_path, data_comparison_path)
