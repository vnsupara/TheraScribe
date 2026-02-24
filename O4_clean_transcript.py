#!/usr/bin/env python3
"""
    python3 clean_transcript.py path/to/input.txt -o output.txt
    python3 clean_transcript.py path/to/input.txt --spell --verbose

By default the cleaned file is written to the current working directory as:
    <input_basename>_cleaned.txt
"""

import re
import sys
import os
import argparse

try:
    from spellchecker import SpellChecker
    SPELLCHECKER_AVAILABLE = True
except Exception:
    SPELLCHECKER_AVAILABLE = False

# Regex patterns
_EPSILON_RE = re.compile(r"<\s*epsilon\s*>", flags=re.IGNORECASE)
_BRACKET_UNINTEL_RE = re.compile(r"\[\s*\[?unintellig(?:ible|igible|ible).*?\]\s*\]", flags=re.IGNORECASE)
_SQUARE_UNINTEL_RE = re.compile(r"\[\s*unintellig(?:ible|igible|ible).*?\]", flags=re.IGNORECASE)
_MULTISPACE_RE = re.compile(r"\s+")
_REPEATED_CHARS_RE = re.compile(r"(.)\1{2,}")
_REPEATED_WORDS_RE = re.compile(r"\b(\w+)(?:\s+\1\b){1,}", flags=re.IGNORECASE)
_PUNCT_SPACE_RE = re.compile(r"\s+([,?.!;:])")
_LEADING_TRAILING_SPACE_RE = re.compile(r"^\s+|\s+$")


def _remove_epsilons(text: str) -> str:
    return _EPSILON_RE.sub(" ", text)


def _normalize_unintelligible(text: str) -> str:
    text = _BRACKET_UNINTEL_RE.sub(" [unintelligible] ", text)
    text = _SQUARE_UNINTEL_RE.sub(" [unintelligible] ", text)
    return text


def _collapse_repeated_chars_in_word(word: str) -> str:
    # Collapse runs of 3+ identical chars to two occurrences (so "soooo" -> "soo")
    return _REPEATED_CHARS_RE.sub(lambda m: m.group(1) * 2, word)


def _collapse_repeated_chars(text: str) -> str:
    tokens = re.split(r"(\W+)", text)  # keep punctuation as separate tokens
    tokens = [_collapse_repeated_chars_in_word(t) if t and t[0].isalnum() else t for t in tokens]
    return "".join(tokens)


def _collapse_repeated_words(text: str) -> str:
    def _rep(m):
        return m.group(1)
    return _REPEATED_WORDS_RE.sub(_rep, text)


def _fix_spacing(text: str) -> str:
    text = _PUNCT_SPACE_RE.sub(r"\1", text)          # remove space before punctuation
    text = _MULTISPACE_RE.sub(" ", text)             # collapse multiple spaces
    text = _LEADING_TRAILING_SPACE_RE.sub("", text)  # trim
    return text


def _spell_correct_text(text: str) -> str:
    if not SPELLCHECKER_AVAILABLE:
        return text
    spell = SpellChecker()
    def _correct_token(tok):
        if not tok.isalpha():
            return tok
        if tok.lower() in spell:
            return tok
        cand = spell.correction(tok)
        if not cand:
            return tok
        if tok[0].isupper():
            return cand.capitalize()
        return cand
    tokens = re.split(r"(\W+)", text)
    tokens = [_correct_token(t) if t and t.isalpha() else t for t in tokens]
    return "".join(tokens)


def clean_transcript(raw_text: str, spell_correct: bool = False) -> str:
    if not raw_text:
        return ""
    text = raw_text
    text = _remove_epsilons(text)
    text = _normalize_unintelligible(text)
    text = _collapse_repeated_chars(text)
    text = _collapse_repeated_words(text)
    text = _fix_spacing(text)
    if spell_correct:
        text = _spell_correct_text(text)
        text = _fix_spacing(text)
    # Collapse long repeated tokens like "la la la la" -> "la la"
    text = re.sub(r"\b(\w+)(?:\s+\1){2,}\b", lambda m: " ".join([m.group(1)] * 2), text, flags=re.IGNORECASE)
    return text


def main():
    parser = argparse.ArgumentParser(description="Clean ASR transcript text file")
    parser.add_argument("input", help="Input transcript text file")
    parser.add_argument("-o", "--output", help="Output file path (default: <cwd>/<basename>_cleaned.txt)")
    parser.add_argument("--spell", action="store_true", help="Run light spell correction (requires pyspellchecker)")
    parser.add_argument("--verbose", action="store_true", help="Print debug info")
    args = parser.parse_args()

    input_path = args.input
    if not os.path.exists(input_path):
        print(f"Input file not found: {input_path}", file=sys.stderr)
        sys.exit(2)

    try:
        with open(input_path, "r", encoding="utf-8") as f:
            raw = f.read()
    except Exception as e:
        print(f"Error reading input file: {e}", file=sys.stderr)
        sys.exit(2)

    if args.verbose:
        print(f"Read {len(raw)} bytes from {input_path}")

    cleaned = clean_transcript(raw, spell_correct=args.spell)

    if args.output:
        out_path = args.output
    else:
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        out_path = os.path.join(os.getcwd(), f"{base_name}_cleaned.txt")

    try:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(cleaned.strip() + "\n")
    except Exception as e:
        print(f"Error writing output file: {e}", file=sys.stderr)
        sys.exit(3)

    if args.verbose:
        print(f"Wrote {os.path.getsize(out_path)} bytes to {out_path}")
    else:
        print(f"Wrote cleaned transcript to: {out_path}")


if __name__ == "__main__":
    main()
