import re


URL_PATTERN = re.compile(r"http\S+|www\S+")
MENTION_PATTERN = re.compile(r"@\w+")
MULTISPACE_PATTERN = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """
    Basic tweet cleaning.
    We keep emojis, punctuation, and casing (important for transformers).
    """
    if text is None:
        return ""

    text = text.strip()
    text = URL_PATTERN.sub("", text)
    text = MENTION_PATTERN.sub("", text)
    text = MULTISPACE_PATTERN.sub(" ", text)

    return text.strip()