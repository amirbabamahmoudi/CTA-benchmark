import pandas as pd
import json
from helpers import split_text



def load_datasets() -> tuple:
    """
    Loads the CSV datasets and returns the test and training dataframes.

    Returns:
        tuple: (test_df, train_df)
            - test_df: DataFrame loaded from 'data/sato_cv_0.csv'
            - train_df: DataFrame concatenated from 'data/sato_cv_1.csv', 'data/sato_cv_2.csv',
                        'data/sato_cv_3.csv', 'data/sato_cv_4.csv'
    """
    test_df = pd.read_csv("data/sato_cv_0.csv")
    csv1 = pd.read_csv("data/sato_cv_1.csv")
    csv2 = pd.read_csv("data/sato_cv_2.csv")
    csv3 = pd.read_csv("data/sato_cv_3.csv")
    csv4 = pd.read_csv("data/sato_cv_4.csv")

    # Concatenate training CSVs and reset the index
    train_df = pd.concat([csv1, csv2, csv3, csv4], axis=0).reset_index(drop=True)
    return test_df, train_df


def compute_max_overlap_for_test_row(test_row: pd.Series, train_df: pd.DataFrame) -> float:
    """
    Computes the maximum overlap between the test row's data and training rows of the same class.

    The overlap is defined as:

        overlap = (number of common tokens) / (number of unique tokens in the test data)

    Args:
        test_row (pd.Series): A row from the test DataFrame.
        train_df (pd.DataFrame): The training DataFrame.

    Returns:
        float: The maximum overlap value found.
    """
    max_overlap = 0.0
    test_class = test_row["class"]
    test_tokens = split_text(test_row["data"])
    test_set = set(test_tokens)

    # If there are no tokens, return 0 to avoid division by zero.
    if not test_set:
        return 0.0

    # Filter the training rows to only those with the same class.
    train_subset = train_df[train_df["class"] == test_class]

    for _, train_row in train_subset.iterrows():
        train_tokens = split_text(train_row["data"])
        train_set = set(train_tokens)
        common_elements = test_set.intersection(train_set)
        overlap = len(common_elements) / len(test_set)
        if overlap > max_overlap:
            max_overlap = overlap
    return max_overlap


def compute_overlap_indices(test_df: pd.DataFrame, train_df: pd.DataFrame) -> dict:
    """
    Computes indices of test DataFrame rows that fall into various overlap categories.

    The categories are defined as:
      - "overlap_100": max_overlap == 1
      - "overlap_60": max_overlap <= 0.6
      - "overlap_30": max_overlap <= 0.3
      - "overlap_10": max_overlap <= 0.1

    Note: A single test row index may appear in more than one category.

    Args:
        test_df (pd.DataFrame): The test dataset.
        train_df (pd.DataFrame): The training dataset.

    Returns:
        dict: A dictionary with keys as category names and values as lists of indices.
    """
    overlap_100 = []
    overlap_60 = []
    overlap_30 = []
    overlap_10 = []

    for idx, test_row in test_df.iterrows():
        max_overlap = compute_max_overlap_for_test_row(test_row, train_df)

        if max_overlap == 1:
            overlap_100.append(idx)
        if max_overlap <= 0.6:
            overlap_60.append(idx)
        if max_overlap <= 0.3:
            overlap_30.append(idx)
        if max_overlap <= 0.1:
            overlap_10.append(idx)

    return {
        "overlap_100": overlap_100,
        "overlap_60": overlap_60,
        "overlap_30": overlap_30,
        "overlap_10": overlap_10
    }


def save_overlap_dict(overlap_dict: dict, filename: str) -> None:
    """
    Saves the overlap dictionary to a JSON file.

    Args:
        overlap_dict (dict): Dictionary containing overlap indices.
        filename (str): Path to the JSON file where the dictionary will be saved.
    """
    with open(filename, "w") as json_file:
        json.dump(overlap_dict, json_file, indent=4)


def main():
    # Load test and training datasets.
    test_df, train_df = load_datasets()

    # Compute the overlap indices.
    overlap_dict = compute_overlap_indices(test_df, train_df)

    # Save the results to a JSON file.
    save_overlap_dict(overlap_dict, "overlap_dict.json")

    print("Overlap indices saved to 'overlap_dict.json'.")


if __name__ == "__main__":
    main()
