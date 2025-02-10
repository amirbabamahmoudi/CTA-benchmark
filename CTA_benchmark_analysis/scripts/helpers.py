def split_text(text: str) -> list:
    """
    Splits the input text into tokens using the delimiters ' ' and ', '.
    This function first normalizes the delimiters to a single space and then splits.

    Args:
        text (str): The input string.

    Returns:
        list: A list of tokens.
    """
    for delimiter in [" ", ", "]:
        text = " ".join(text.split(delimiter))
    tokens = text.split()
    return tokens


SINGLE_COLUMN_PROMPT = """
Your task is to classify a given column with only one of the following types (separated by commas):
.
"""

# Prompt for multi-column tables (few-shot example)
MULTI_COLUMN_PROMPT_FEW = """
Your task is to classify a given column with only one of the following types (separated by commas):
'{{COLUMN_TYPES}}'
If you are not able to classify a column, answer with "Not Clear."

Example:
Column 1: HAKALA Kia-Emilia HAKALA Kia-Emilia HOIKKANEN Emilia
Column 2: FIN FIN FIN FIN FIN FIN
Column 3: F F F F M M
Column 4: L L L L L L

Column types:
Column 1: name
Column 2: country
Column 3: sex
Column 4: class
"""