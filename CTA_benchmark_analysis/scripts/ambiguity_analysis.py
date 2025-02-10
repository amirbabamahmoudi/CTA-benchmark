import pandas as pd
from helpers import split_text




def load_dataset(filepath: str) -> pd.DataFrame:
    """
    Loads a CSV file into a DataFrame.

    Args:
        filepath (str): Path to the CSV file.

    Returns:
        pd.DataFrame: The loaded DataFrame.
    """
    return pd.read_csv(filepath)


def run_ambiguity_analysis(df: pd.DataFrame, max_rows: int = 2000) -> pd.DataFrame:
    """
    Performs ambiguity analysis by comparing each row's 'data' field to every other row.

    For each row (index i), if all tokens (after tokenization) are found in a different row (index j)
    and the two rows have different class labels (with additional conditions), then:

      - Row i is added (with flag 0) the first time a match is found.
      - Each unique class from a matching row j is also added (with flag 1).

    The analysis stops early if the accumulated results exceed max_rows.

    Args:
        df (pd.DataFrame): DataFrame with at least the columns ['table_id', 'data', 'class'].
        max_rows (int): Maximum number of result rows to accumulate.

    Returns:
        pd.DataFrame: DataFrame with the ambiguous rows and a 'flag' column.
    """
    results = []
    n = len(df)

    for i in range(n):
        flag_set = False  # Indicates if row i has already been added.
        added_classes = []  # Keeps track of classes added from comparisons with row i.

        # Process row i
        data_i = df.at[i, "data"].lower()
        tokens_i = split_text(data_i)
        # Create a list of tokens that are entirely digits.
        digit_tokens = [s for s in tokens_i if s.isdigit()]

        # Compare row i with every other row j.
        for j in range(n):
            if i == j:
                continue

            data_j = df.at[j, "data"].lower()
            tokens_j = split_text(data_j)

            set_i = set(tokens_i)
            set_j = set(tokens_j)
            common_tokens = set_i.intersection(set_j)

            # Check if all tokens in row i appear in row j,
            # the classes are different,
            # row i does not consist solely of digit tokens,
            # and row i's class is not one of the excluded types.
            if (len(common_tokens) == len(set_i) and
                    df.at[i, "class"] != df.at[j, "class"] and
                    len(tokens_i) != len(digit_tokens) and
                    df.at[i, "class"] not in {"rank", "position", "result"}):

                # Add row i once (with flag 0) when the first match is found.
                if not flag_set:
                    results.append({
                        "table_id": df.at[i, "table_id"],
                        "data": df.at[i, "data"],
                        "class": df.at[i, "class"],
                        "flag": 0
                    })
                    flag_set = True

                # For each unique class of row j, add row j with flag 1.
                if df.at[j, "class"] not in added_classes:
                    results.append({
                        "table_id": df.at[j, "table_id"],
                        "data": df.at[j, "data"],
                        "class": df.at[j, "class"],
                        "flag": 1
                    })
                    added_classes.append(df.at[j, "class"])

        # Stop the analysis if we exceed the specified maximum number of result rows.
        if len(results) > max_rows:
            print(f"Reached maximum row limit at index {i}. Stopping analysis.")
            break

    # Convert the results list into a DataFrame.
    return pd.DataFrame(results, columns=["table_id", "data", "class", "flag"])


def save_dataframe_to_zip(df: pd.DataFrame, zip_filename: str, csv_filename: str) -> None:
    """
    Saves a DataFrame as a compressed ZIP file containing a CSV file.

    Args:
        df (pd.DataFrame): The DataFrame to save.
        zip_filename (str): The filename of the output ZIP file.
        csv_filename (str): The name of the CSV file inside the ZIP archive.
    """
    compression_opts = dict(method='zip', archive_name=csv_filename)
    df.to_csv(zip_filename, index=False, compression=compression_opts)
    print(f"Data saved to {zip_filename} as {csv_filename}.")


def main():
    # Load the dataset (assumed to be at 'data/sato_cv_0.csv').
    dataset_path = "data/sato_cv_0.csv"
    df = load_dataset(dataset_path)

    # Run the ambiguity analysis.
    ambiguous_df = run_ambiguity_analysis(df, max_rows=2000)

    # Save the results as a compressed CSV file.
    save_dataframe_to_zip(ambiguous_df, "same_column_dif_type.zip", "same_column_dif_type.csv")


if __name__ == "__main__":
    main()
