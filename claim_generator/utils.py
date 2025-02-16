import json
import time
from pathlib import Path
from typing import List, Dict, Any

import requests
from openai import APIConnectionError, Timeout, RateLimitError, APIError, APITimeoutError


def save_claims_to_json(
        items: List[Dict[str, Any]],
        output_path: Path
) -> None:
    """
    Save the items (with claims) to the specified JSON path.
    """
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(items, f, indent=2, ensure_ascii=False)


def retry_with_exponential_backoff(
        func,
        max_retries=5,
        initial_delay=2.0,
        backoff_factor=2.0,
        retry_exceptions=(
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                APIConnectionError,
                Timeout,
                RateLimitError,
                APITimeoutError,
                APIError
        ),
        logger=None
):
    """
    Decorator-like function for retrying an API call with exponential backoff.
    `func` should be a callable that does the network request.

    :param func: The API-calling function you want to wrap.
    :param max_retries: Maximum number of times to attempt the function.
    :param initial_delay: Initial sleep time between retries.
    :param backoff_factor: Growth factor for delay between each attempt.
    :param retry_exceptions: Tuple of exceptions that trigger a retry.
    :param logger: Optional logger function for printing/warning.

    Usage:
        result = retry_with_exponential_backoff(my_request_func)(**kwargs)
    """

    def wrapper(*args, **kwargs):
        delay = initial_delay
        for attempt in range(1, max_retries + 1):
            try:
                return func(*args, **kwargs)
            except retry_exceptions as e:
                if attempt == max_retries:
                    # Exhausted all retries; re-raise
                    raise
                if logger:
                    logger(f"[WARN] Attempt {attempt} failed with '{e}'. Retrying in {delay} seconds...")
                time.sleep(delay)
                delay *= backoff_factor

    return wrapper
