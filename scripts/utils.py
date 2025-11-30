"""Utility functions for data processing and output generation."""

import pandas as pd
from scipy import stats
from scipy.optimize import brentq

# Load labels and questions once when module is imported
LABELS_PATH = "../data/labels.csv"
QUESTIONS_PATH = "../data/questions.csv"
labels = pd.read_csv(LABELS_PATH)
questions = pd.read_csv(QUESTIONS_PATH)

def get_value_for_label(item_name: str, label_text: str) -> int:
    """
    Get the numeric value for a given answer label.

    Args:
        item_name: The question/item name (e.g., 'consent', 'gender')
        label_text: The answer text to look up (e.g., 'I agree to participate in the study')

    Returns:
        The column number as int

    Example:
        >>> get_value_for_label('consent', 'I agree to participate in the study')
        '1'
    """
    item_row = labels[labels['item'] == item_name]
    if item_row.empty:
        raise ValueError(f"Item '{item_name}' not found in labels.csv")

    # Search through all columns to find the matching label
    for col in item_row.columns:
        if col == 'item':
            continue
        value = item_row[col].iloc[0]
        if pd.notna(value) and value == label_text:
            return col

    raise ValueError(f"Label '{label_text}' not found for item '{item_name}'")


def get_label_for_value(item_name: str, value: str | int) -> str:
    """
    Get the answer label for a given numeric value.

    Args:
        item_name: The question/item name (e.g., 'consent', 'gender')
        value: The numeric value to look up (e.g., '1', 1, '2', 2)

    Returns:
        The label text corresponding to that value

    Example:
        >>> get_label_for_value('consent', '1')
        'I agree to participate in the study'
    """
    item_row = labels[labels['item'] == item_name]
    if item_row.empty:
        raise ValueError(f"Item '{item_name}' not found in labels.csv")

    # Convert value to string to match column names
    value_str = str(value)

    if value_str not in item_row.columns:
        raise ValueError(f"Value '{value}' is not a valid column for item '{item_name}'")

    label_text = item_row[value_str].iloc[0]

    if pd.isna(label_text):
        raise ValueError(f"No label found for value '{value}' in item '{item_name}'")

    return label_text


def get_question_statement(item_name: str) -> str:
    """
    Get the question statement for a given item.

    Args:
        item_name: The question/item name (e.g., 'consent', 'gender', 'ATI_1')

    Returns:
        The question statement text

    Example:
        >>> get_question_statement('consent')
        'Consent Statement\\n\\nI confirm that:...'
        >>> get_question_statement('ATI_1')
        'I like to occupy myself in greater detail with technical systems.'
    """
    item_row = questions[questions['item'] == item_name]
    if item_row.empty:
        raise ValueError(f"Item '{item_name}' not found in questions.csv")

    question_text = item_row['question_statement'].iloc[0]

    if pd.isna(question_text):
        raise ValueError(f"No question statement found for item '{item_name}'")

    return question_text

def apa_p(p, sig_stars=False):
    """
    Format p-value in APA style (no leading zero).

    Args:
        sig_stars: Returns significance value with '*' stars as per APA formatting conventions.

    Returns:
        The formatted p-value.

    """
    def apa_sig(p):
        """Return significance stars."""
        if p < 0.001:
            return "***"
        elif p < 0.01:
            return "**"
        elif p < 0.05:
            return "*"
        else:
            return ""

    if p < 0.001:
        output =  "< .001"
    else:
        output =  f"{p:.3f}".replace("0.", ".")

    if sig_stars:
        output += ' ' + apa_sig(p)

    return output
