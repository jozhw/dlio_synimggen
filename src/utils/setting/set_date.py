import re


def set_date(file_path: str) -> str:
    regex_date_pattern: str = r"\b\d{4}-\d{2}-\d{2}\b"

    date: str = re.findall(regex_date_pattern, file_path)[0]

    return date
