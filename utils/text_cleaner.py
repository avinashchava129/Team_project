import re

def clean_text(text: str) -> str:
    """
    Cleans raw text extracted from a PDF resume.

    Responsibilities:
    - Removes multiple spaces/tabs.
    - Removes unnecessary newlines (e.g. paragraph wrapping within sentences).
    - Preserves useful formatting like bullet points, lists, and section headers.
    - Normalizes vertical whitespace (max 2 consecutive newlines).

    Args:
        text: Raw text string.

    Returns:
        Cleaned and normalized text string.
    """
    if not text:
        return ""

    # 1. Normalize line endings (CRLF -> LF)
    text = text.replace("\r\n", "\n")

    # 2. Split into lines and clean horizontal spacing for each line
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]

    cleaned_lines = []
    i = 0
    n = len(lines)
    
    while i < n:
        line = lines[i]

        # If it's an empty line, preserve it as a paragraph/section break
        if not line:
            cleaned_lines.append("")
            i += 1
            continue

        # Heuristic to join split lines:
        # If the current line does not end with sentence-ending punctuation (., :, ;, !, ?)
        # AND the next line is not empty AND starts with a lowercase letter,
        # we treat it as an unnecessary newline and merge them.
        while (i + 1 < n and 
               lines[i + 1] and 
               not line.endswith((".", ":", ";", "!", "?", "•", "-", "*")) and 
               lines[i + 1][0].islower()):
            line = line + " " + lines[i + 1]
            i += 1

        cleaned_lines.append(line)
        i += 1

    # 3. Reconstruct text by joining lines
    cleaned_text = "\n".join(cleaned_lines)

    # 4. Normalize vertical whitespace: collapse 3+ consecutive newlines into 2 (double newline)
    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)

    # 5. Final horizontal space normalization (just in case)
    cleaned_text = re.sub(r" {2,}", " ", cleaned_text)

    return cleaned_text.strip()
