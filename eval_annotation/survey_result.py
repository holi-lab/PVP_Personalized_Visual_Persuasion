import pandas as pd
import openpyxl

# Configuration: Declare paths at the top
batch_num = 1  # Adjust the batch number here if needed
base_path = "../../../data"

# Paths for different input files
path_match = f"{base_path}/match/final/batch{batch_num}.csv"
path_description = f"{base_path}/description/batch{batch_num}.csv"
path_score = f"{base_path}/score/batch{batch_num}_score.csv"
path_piping = f"{base_path}/piping/piping{batch_num}.xlsx"
path_duplicated = f"{base_path}/duplicated/batch{batch_num}/"

# Output path for the final result
output_path = f"{base_path}/final_result_batch{batch_num}.csv"

# Load data
df_match = pd.read_csv(path_match)
df_description = pd.read_csv(path_description)
df_score = pd.read_csv(path_score)

# Drop unnecessary columns
df_score.drop(["Unnamed: 0", "Unnamed: 0.1"], inplace=True, axis=1)

# Load piping data and filter relevant columns
piping = pd.read_excel(path_piping)
piping_filtering = piping[["Message", "habit type", "habit reverse", "Topic"]]

# Add habit information from piping to df_score
for idx, row in piping_filtering.iterrows():
    df_score.loc[df_score["Gubun"] == row["Message"], ["habit type", "habit reverse", "Topic"]] = row[["habit type", "habit reverse", "Topic"]].values

# Add gubun column to df_description based on df_match
def gubun_columns(df_description: pd.DataFrame, df_match: pd.DataFrame) -> pd.DataFrame:
    df_description["gubun"] = None
    for i, row in df_description.iterrows():
        matching_row = df_match[df_match["eng"] == row["message"]]
        if not matching_row.empty:
            df_description.at[i, "gubun"] = matching_row["Gubun"].values[0]
    return df_description

df_description_gubun = gubun_columns(df_description, df_match)

# Function to match old_number with new_number
def match_new_number(data, df_duplicated):
    filtered_data = data[data['old_number'].isin(df_duplicated['old_number'])]
    for index, row in filtered_data.iterrows():
        old_number = row['old_number']
        new_number = df_duplicated[df_duplicated['old_number'] == old_number]['new_number'].values[0]
        filtered_data.loc[filtered_data['old_number'] == old_number, 'new_number'] = new_number
    return filtered_data

# Process each gubun to generate description_batch
all_results = []
for gubun in range(1, 61):
    data_dalle = df_description_gubun[(df_description_gubun["method"] == "dalle") & (df_description_gubun["gubun"] == gubun)]
    data_google = df_description_gubun[(df_description_gubun["method"] == "google") & (df_description_gubun["gubun"] == gubun)]
    df_duplicated = pd.read_csv(f"{path_duplicated}{gubun}.csv", encoding="cp949")
    df_duplicated_dalle = df_duplicated[df_duplicated["method"] == "dalle"]
    df_duplicated_google = df_duplicated[df_duplicated["method"] == "google"]
    
    dalle_df = match_new_number(data_dalle, df_duplicated_dalle)
    google_df = match_new_number(data_google, df_duplicated_google)
    
    result = pd.concat([dalle_df, google_df])
    all_results.append(result)

description_batch = pd.concat(all_results, ignore_index=True)

# Function to create the final dataset for each gubun
def make_final_dataset(gubun: int, df_score: pd.DataFrame, df_description: pd.DataFrame) -> pd.DataFrame:
    df_batch_gubun = df_score[df_score["Gubun"] == gubun]
    df_des_gubun = df_description[df_description["gubun"] == gubun]
    
    columns = ["Batch", "Gubun", "NO", "Gender", "Age", "Image_Num", 
               "Topic", "Message", "Strategy", "Method", "Pos_Neg", 
               "Description", "Score", "Habit", "Habit_type", 
               "Habit_reverse", "Values"]
    final_df = pd.DataFrame(columns=columns)
    rows_to_add = []
    
    num_columns = df_batch_gubun.filter(regex='^Q1').dropna(axis=1).shape[1]
    
    for index, row in df_batch_gubun.iterrows():
        for num in range(1, num_columns + 1):
            image_num = num
            number = row["NO"]
            Gubun = row["Gubun"]
            gender = row["SQ1"]
            age = row["SQ2"]
            habit = row["Q2_1"]
            habit_type = row["habit type"]
            habit_reverse = row["habit reverse"]
            Topic = row["Topic"]
            score = row[f"Q1_{num}"]
            method = df_des_gubun[df_des_gubun["new_number"] == num]["method"].values[0]
            message = df_des_gubun[df_des_gubun["new_number"] == num]["message"].values[0]
            strategy = df_des_gubun[df_des_gubun["new_number"] == num]["strategy"].values[0]
            pos_neg = df_des_gubun[df_des_gubun["new_number"] == num]["pos_neg"].values[0]
            description = df_des_gubun[df_des_gubun["new_number"] == num]["description"].values[0]
            
            values_dict = {col: row[col] for col in df_score.columns[-23:-3] if col in row}
            values_str = str(values_dict)
            
            new_row = {
                "Batch": batch_num,
                "Gubun": Gubun,
                "Topic": Topic,
                "NO": number,
                "Gender": gender,
                "Age": age,
                "Image_Num": image_num,
                "Method": method,
                "Message": message,
                "Strategy": strategy,
                "Pos_Neg": pos_neg,
                "Description": description,
                "Score": score,
                "Habit": habit,
                "Habit_type": habit_type,
                "Habit_reverse": habit_reverse,
                "Values": values_str
            }
            
            rows_to_add.append(new_row)
    
    final_df = pd.concat([final_df, pd.DataFrame(rows_to_add)], ignore_index=True)
    
    return final_df

# Process each gubun excluding some values
list_result = []
excluded = [] ## if being some excluded value
for gubun in [i for i in range(1, 61) if i not in excluded]:
    print("Processing Gubun:", gubun)
    result_gubun = make_final_dataset(gubun, df_score, description_batch)
    list_result.append(result_gubun)

# Final result
final_result = pd.concat(list_result, ignore_index=True)

# Save the final result
final_result.to_csv(output_path, index=False)