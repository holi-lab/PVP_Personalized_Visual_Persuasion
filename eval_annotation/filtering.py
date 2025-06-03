import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict

# Path definitions
batch_num = 1  # You can change the batch number here
path_annotating = f"../../../data/batch/Batch{batch_num}.xlsx"
path_repeat = f"../../../data/duplicated/batch{batch_num}/"

class DataAnalyzer:
    def __init__(self, gubun: int):
        self.gubun = gubun
        self.data_annotating = pd.read_excel(path_annotating)
        self.df_repeat, self.df_annotation = self.prepare_data()

    # Method to prepare the data
    def prepare_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        df_repeat = pd.read_csv(f"{path_repeat}{self.gubun}.csv", encoding="cp949")
        df_annotation = self.data_annotating[self.data_annotating["Gubun"] == self.gubun]
        return df_repeat, df_annotation

    # Method to get duplicate images
    def duplicate_image(self) -> Tuple[List[int], List[int]]:
        df = self.df_repeat
        duplicate_row = df[df["repeat"] != 0]
        original_image_index = duplicate_row["new_number"].astype(int).tolist()
        duplicate_image_index = duplicate_row["repeat"].astype(int).tolist()
        return original_image_index, duplicate_image_index

    # Method to analyze the gap in scores
    def analyze_gap(self) -> List[int]:
        list_key = []
        original_index, duplicate_index = self.duplicate_image()
        score_gap_list, total_score = self.score_gap(original_index, duplicate_index)
        filtered_total_score = {k: v for k, v in total_score.items() if v >= 2}
        for i in filtered_total_score.keys():
            list_key.append(i)
        return list_key

    # Method to calculate the gap between original and duplicate scores
    def score_gap(self, index_original: List[int], index_duplication: List[int]) -> Tuple[List[Dict[int, float]], Dict[int, int]]:
        score_gap_list = []
        total_score = {}
        
        for org, dup in zip(index_original, index_duplication):
            original_col = "Q1_" + str(org)
            duplication_col = "Q1_" + str(dup)
            gap = abs(self.df_annotation[original_col] - self.df_annotation[duplication_col])
            
            gap_dict = {idx: gap_val for idx, gap_val in zip(gap.index, gap.values)}
            score_gap_list.append(gap_dict)
            
            for key, value in gap_dict.items():
                if key not in total_score:
                    total_score[key] = 0
                if value >= 2:
                    total_score[key] += 1
        return score_gap_list, total_score

    # Method to extract Big5 tags from annotations
    def Big5_tag(self) -> pd.DataFrame:
        Big5_annotating = self.df_annotation.filter(regex='^Q2_2')
        Big5_annotating.index = self.df_annotation["NO"].values
        result_Big5 = Big5_annotating.dropna(axis=1)
        return result_Big5

    # Method to calculate Big5 score consistency
    def Big5_score(self, df: pd.DataFrame) -> List[int]:
        list_Extraversion = [1, 6]
        list_Agreeableness = [2, 7]
        list_Conscientiousness = [3, 8]
        list_Neuroticism = [4, 9]
        list_Openness = [5, 10]

        df_scores = pd.DataFrame(index=df.index)

        def reverse_score(value):
            return 5 - value

        def same_boundary(val1, val2):
            return abs(val1 - val2) <= 2

        df_scores['Extraversion_SameBoundary'] = df.apply(lambda row: same_boundary(reverse_score(row[f'Q2_2_{list_Extraversion[0]}']), row[f'Q2_2_{list_Extraversion[1]}']), axis=1)
        df_scores['Agreeableness_SameBoundary'] = df.apply(lambda row: same_boundary(row[f'Q2_2_{list_Agreeableness[0]}'], reverse_score(row[f'Q2_2_{list_Agreeableness[1]}'])), axis=1)
        df_scores['Conscientiousness_SameBoundary'] = df.apply(lambda row: same_boundary(reverse_score(row[f'Q2_2_{list_Conscientiousness[0]}']), row[f'Q2_2_{list_Conscientiousness[1]}']), axis=1)
        df_scores['Neuroticism_SameBoundary'] = df.apply(lambda row: same_boundary(reverse_score(row[f'Q2_2_{list_Neuroticism[0]}']), row[f'Q2_2_{list_Neuroticism[1]}']), axis=1)
        df_scores['Openness_SameBoundary'] = df.apply(lambda row: same_boundary(reverse_score(row[f'Q2_2_{list_Openness[0]}']), row[f'Q2_2_{list_Openness[1]}']), axis=1)
        
        false_index = df_scores[(df_scores == False).any(axis=1)].index.tolist()
        return false_index

    # Main method to perform the Big5 analysis
    def big5_main(self) -> List[int]:
        Big5_annotating = self.Big5_tag()
        return self.Big5_score(Big5_annotating)

    # Main method to analyze score gap and Big5 consistency
    def main(self) -> List[int]:
        score_gap = self.analyze_gap()
        big5_gap = self.big5_main()
        print("Score gap:", score_gap)
        print("Big5 gap:", big5_gap)
        result = list(set(big5_gap) & set(score_gap))
        return result


class ImageAnalysis:
    def __init__(self, gubun: int, path_annotating: str):
        self.gubun = gubun
        self.data_annotating = pd.read_excel(path_annotating)
        self.df = self.prepare_data()
        self.result_image = None

    # Method to prepare data
    def prepare_data(self) -> pd.DataFrame:
        df_annotation = self.data_annotating[self.data_annotating["Gubun"] == self.gubun]
        return df_annotation

    # Method to extract image tags
    def image_tag(self) -> pd.DataFrame:
        image_annotating = self.df.filter(regex='^Q1_')
        image_annotating.index = self.df["NO"].values
        self.result_image = image_annotating.dropna(axis=1)
        return self.result_image

    # Method to plot the scores for each image
    def plot_scores(self) -> None:
        if self.result_image is None:
            self.image_tag()
        
        for index, row in self.result_image.iterrows():
            plt.figure(figsize=(5, 2))
            plt.plot(row.values, 'o-', markersize=4, color='orange')
            plt.title(f'Scores for NO: {index}')
            plt.xlabel('Index')
            plt.ylabel('Score')
            plt.tight_layout()
            plt.show()
    
    # Method to calculate variance in scores
    def calculate_variance(self, variance_threshold=0.1) -> List[int]:
        if self.result_image is None:
            self.image_tag()
        
        variance = self.result_image.var(axis=1, ddof=1)
        print("Variance of each annotator's responses:")
        print(variance)
        
        low_variance = variance[variance < variance_threshold]
        if not low_variance.empty:
            print(f"\nResponses with variance lower than {variance_threshold}:")
            print(low_variance)
        else:
            print(f"\nNo responses with variance lower than {variance_threshold}.")
        
        high_variance_index = variance[variance >= variance_threshold].index.tolist()
        print("\nResponses with variance above the threshold:")
        print(variance[variance >= variance_threshold])
        
        return high_variance_index

    # Main method for image analysis
    def main(self) -> List[int]:
        result = self.calculate_variance()
        # Uncomment below line to plot scores if needed
        # self.plot_scores()
        return result


if __name__ == "__main__":
    # Data analysis part
    list_out = []
    for gubun in range(1, 61, 1):
        print(f"Processing Gubun {gubun}")
        analyzer = DataAnalyzer(gubun=gubun)
        results = analyzer.main()
        list_out.extend(results)

    # Image analysis part
    index_list = []
    for gubun in range(1, 61, 1):
        analysis = ImageAnalysis(gubun, path_annotating)
        result = analysis.main()
        index_list.extend(result)