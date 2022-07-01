import pandas as pd
from . import cleaning
from tqdm import tqdm
import os
import ast
tqdm.pandas()


def get_cleaned_data(df):
    """Given the dataframe, return a cleaned version with columns [cleaned_q1, cleaned_q2, is_duplicate]
    Params: Dataframe with columns ['question1', 'question2', 'is_duplicate']
    Returns: pd.DataFrame
    """
    path = "../data/cleaned_data.csv"
    if not os.path.exists(path):
        print("Cleaning data...")
        df["cleaned_q1"] = df["question1"].progress_apply(cleaning.clean)
        df["cleaned_q2"] = df["question2"].progress_apply(cleaning.clean)
        print("Finished cleaning")
        clean_df = df[["cleaned_q1", "cleaned_q2", "is_duplicate"]]
        clean_df.to_csv("../data/cleaned_data.csv", index=False)
        return clean_df
    else:
        clean_df = pd.read_csv(path)
        clean_df["cleaned_q1"] = clean_df["cleaned_q1"].apply(lambda x: ast.literal_eval(x))
        clean_df["cleaned_q2"] = clean_df["cleaned_q2"].apply(lambda x: ast.literal_eval(x))
        return clean_df
    
def get_all_questions(df):
    """Creates a series of all the cleaned questions to be used for further processing
    Returns: pd.Series
    """
    if not "cleaned_q1" in df.columns or not "cleaned_q2" in df.columns:
        print("Please pass in a datframe that already has it's text preprocessed")
        return
    all_questions = pd.Series(pd.concat((df["cleaned_q1"], df["cleaned_q2"])))
    all_questions = all_questions.dropna()
    return all_questions
    

    
    
    
