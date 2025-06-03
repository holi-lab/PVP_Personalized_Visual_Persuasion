import pandas as pd
from typing import List, Tuple
import openpyxl

# Configuration
batch_num = 1  # You can change the batch number here
path_annotating = f"../../../data/batch/batch_good/Batch{batch_num}.xlsx"
path_repeat = f"../../../data/duplicated/batch{batch_num}/"
output_path = f"../../../data/not_duplicated/Batch{batch_num}.csv"
gubun_range = range(1, 61, 1)  # You can adjust the range of gubun as needed

# Define Reducer class
class Reducer:
    def __init__(self, gubun: int):
        self.gubun = gubun
        self.data_annotating = pd.read_excel(path_annotating)
        self.df_repeat, self.df_annotation = self.prepare_data()
        self.duplicate_indices = []

    # Prepare the data
    def prepare_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        df_repeat = pd.read_csv(f"{path_repeat}{self.gubun}.csv", encoding="cp949")
        df_annotation = self.data_annotating[self.data_annotating["Gubun"] == self.gubun]
        return df_repeat, df_annotation

    # Find duplicate image indices
    def duplicate_image(self) -> Tuple[List[int], List[int]]:
        df = self.df_repeat
        duplicate_row = df[df["repeat"] != 0]
        original_image_index = duplicate_row["new_number"].astype(int).tolist()
        duplicate_image_index = duplicate_row["repeat"].astype(int).tolist()
        return original_image_index, duplicate_image_index
    
    # Get duplicate indices
    def get_duplicate_indices(self) -> List[int]:
        _, self.duplicate_indices = self.duplicate_image()
        return self.duplicate_indices

    # Function to reduce duplicates in annotation
    def reducing_function(self) -> pd.DataFrame:
        # Create a copy to avoid modifying the original data
        score_annotating = self.df_annotation.copy()
        
        # Set duplicate columns to NaN
        for num in self.duplicate_indices:
            col_name = f"Q1_{num}"
            if col_name in score_annotating.columns:
                score_annotating[col_name] = pd.NA
        
        return score_annotating

    # Main function to handle the full process
    def main(self) -> pd.DataFrame:
        self.get_duplicate_indices()
        reduced_data = self.reducing_function()
        return reduced_data

# Initialize a list to store all dataframes
all_dataframes = []

# Process each gubun and store results in list
for gubun in gubun_range:
    reducer_gubun = Reducer(gubun=gubun)
    reduced_df = reducer_gubun.main()
    all_dataframes.append(reduced_df)

# Concatenate all dataframes into one
final_dataframe = pd.concat(all_dataframes, ignore_index=True)

# Print the final dataframe (optional)
print(final_dataframe)

# Save the final dataframe to a CSV file
final_dataframe.to_csv(output_path, index=False)