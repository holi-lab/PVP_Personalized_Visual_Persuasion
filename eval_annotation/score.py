import pandas as pd

# Configuration
batch_num = 1  # You can change the batch number here
base_path = "../../../data"
path_annotating = f"{base_path}/not_duplicated/0727/Batch{batch_num}.csv"
output_path = f"{base_path}/score/batch{batch_num}_score.csv"

# SurveyScorer class to process and score different survey types
class SurveyScorer:
    def __init__(self):
        self.data = pd.read_csv(path_annotating)
        self.big5_data = self.Big5_tag()  # Extract Big5 data
        self.pvq21_data = self.pvq21_tag()  # Extract PVQ21 data
        self.mfq30_data = self.mfq30_tag()  # Extract MFQ30 data

    # Extract Big5 survey data
    def Big5_tag(self) -> pd.DataFrame:
        Big5_annotating = self.data.filter(regex='^Q2_2')
        Big5_annotating.index = self.data["NO"].values
        result_Big5 = Big5_annotating.dropna(axis=1)
        return result_Big5

    # Extract PVQ21 survey data
    def pvq21_tag(self) -> pd.DataFrame:
        pvq21_annotating = self.data.filter(regex='^Q2_3')
        pvq21_annotating.index = self.data["NO"].values
        result_pvq21 = pvq21_annotating.dropna(axis=1)
        return result_pvq21

    # Extract MFQ30 survey data
    def mfq30_tag(self) -> pd.DataFrame:
        mfq30_annotating = self.data.filter(regex='^Q2_4')
        mfq30_annotating.index = self.data["NO"].values
        result_mfq30 = mfq30_annotating.dropna(axis=1)
        return result_mfq30

    # Score Big5 (BFI-10)
    def score_bfi_10(self) -> pd.DataFrame:
        traits = {
            'Extraversion': [0, 5],
            'Agreeableness': [1, 6],
            'Conscientiousness': [2, 7],
            'Neuroticism': [3, 8],
            'Openness': [4, 9]
        }
        
        reversed_items = [0, 2, 3, 4, 6]
        
        all_results = []
        for idx in range(len(self.big5_data)):
            scores = self.big5_data.iloc[idx].tolist()
            results = {}
            for trait, indices in traits.items():
                total_score = 0
                for index in indices:
                    score = scores[index]
                    if index in reversed_items:
                        score = 6 - score
                    total_score += score
                results[trait] = total_score
            all_results.append(results)
        
        results_df = pd.DataFrame(all_results)
        return results_df
    
    # Score PVQ21
    def score_pvq_21(self) -> pd.DataFrame:
        index_items = {
            'Conformity': [6, 15],
            'Tradition': [8, 19],
            'Benevolence': [11, 17],
            'Universalism': [2, 7, 18],
            'Self-Direction': [0, 10],
            'Stimulation': [5, 14],
            'Hedonism': [9, 20],
            'Achievement': [3, 12],
            'Power': [1, 16],
            'Security': [4, 13],
        }

        all_results = []
        for idx in range(len(self.pvq21_data)):
            results = {}
            for index, items in index_items.items():
                mean_score = self.pvq21_data.iloc[idx, items].mean()
                results[index] = mean_score
            all_results.append(results)

        results_df = pd.DataFrame(all_results)
        return results_df

    # Score MFQ30
    def score_moral_30(self) -> pd.DataFrame:
        categories = {
            'Harm/Care': [0, 6, 11, 16, 22, 27],
            'Fairness/Reciprocity': [1, 7, 12, 17, 23, 28],
            'In-group/Loyalty': [2, 8, 13, 18, 24, 29],
            'Authority/Respect': [3, 9, 14, 19, 25, 30],
            'Purity/Sanctity': [4, 10, 15, 20, 26, 31]
        }

        all_results = []
        for idx in range(len(self.mfq30_data)):
            results = {}
            for category, indices in categories.items():
                total_score = sum(self.mfq30_data.iloc[idx, i] for i in indices)
                results[category] = total_score
            all_results.append(results)

        results_df = pd.DataFrame(all_results)
        return results_df

    # Concatenate all survey scores into one DataFrame
    def concat_scores(self) -> pd.DataFrame:
        # Get and score Big5 data
        big5_scores = self.score_bfi_10().reset_index(drop=True)
        self.data = self.data.reset_index(drop=True)
        result = pd.concat([self.data, big5_scores], axis=1)

        # Get and score PVQ21 data
        pvq21_scores = self.score_pvq_21().reset_index(drop=True)
        result = pd.concat([result, pvq21_scores], axis=1)

        # Get and score MFQ30 data
        mfq30_scores = self.score_moral_30().reset_index(drop=True)
        result = pd.concat([result, mfq30_scores], axis=1)

        return result

# Example usage of the SurveyScorer class
if __name__ == "__main__":
    scorer = SurveyScorer()

    # Concatenate scores
    final_data = scorer.concat_scores()

    # Save the final data to a CSV file
    final_data.to_csv(output_path, index=False)

    # Optionally, print the final dataframe
    print(final_data)