import os
import json
from typing import Dict, List, Union
import ast


def parse_string_to_obj(text: str) -> Union[dict, list, None]:
    """
    Safely parse a string into a Python list or dictionary.
    Handles JSON or Python literal formats, and strips optional Markdown code block markers.
    
    Args:
        text (str): The input string to parse.

    Returns:
        Union[dict, list, None]: Parsed object if successful, otherwise None.
    """
    # Remove Markdown code block markers
    text = text.strip().lstrip('```python').lstrip('```json').rstrip('```').strip()
    # Try JSON parsing
    try:
        return json.loads(text)
    except:
        pass
    # Try Python literal evaluation
    try:
        return ast.literal_eval(text)
    except:
        pass
    return None


def check_match_ids(lst):
    for i in range(len(lst)):
        if lst[i][0] != i + 1:
            return False
    return True


# ANSI color codes for terminal output formatting
GREEN = '\033[92m'
RED = '\033[91m'
RESET = '\033[0m'

def print_green(*args):
    print(GREEN, *args, RESET)

def print_red(*args):
    print(RED, *args, RESET)