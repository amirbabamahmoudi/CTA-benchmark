import openai
from helpers import SINGLE_COLUMN_PROMPT
# --- Prompt Constants ---

# Prompt for single-column type classification



# --- OpenAI API Call Function ---

def call_openai_for_column_type(prompt: str, column_data: str, model: str = "gpt-3.5-turbo") -> str:
    """
    Calls the OpenAI ChatCompletion API with a given prompt and column data to get a predicted column type.

    Args:
        prompt (str): The instructions to the model.
        column_data (str): The data of the column to classify.
        model (str): The model to use (default: "gpt-3.5-turbo").

    Returns:
        str: The model's predicted column type.
    """
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
        {"role": "user", "content": column_data}
    ]

    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
    )

    # Return the response message (strip any extra whitespace)
    return response['choices'][0]['message']['content'].strip()


# --- Example Usage ---

def main():
    # Make sure your OpenAI API key is set in your environment or uncomment and set it here.
    # openai.api_key = "YOUR_OPENAI_API_KEY"

    # Example: classify a single column.
    sample_column_data = "Column 1: Colony Financial, Inc. Colony Financial, Inc. Colony Financial, Inc."
    predicted_type = call_openai_for_column_type(SINGLE_COLUMN_PROMPT, sample_column_data)

    print("Predicted Column Type:")
    print(predicted_type)


if __name__ == "__main__":
    main()
