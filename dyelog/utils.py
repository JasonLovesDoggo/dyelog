from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path
from typing import List

from dyelog.settings import settings


class PatternMatcher:
    def __init__(self, file: Path | None = None) -> None:
        # Define the character set mappings
        self.char_sets = {
            "A-F": set("ABCDEF"),
            "G-M": set("GHIJKLM"),
            "N-T": set("NOPQRST"),
            "U-Z": set("UVWXYZ"),
        }
        self.preprocess_words(file=file)  # type: ignore

    def preprocess_words(
        self,
        min_length: int | None = None,
        file: Path | None = None,
    ) -> None:
        """
        Preprocess the word list and store by length for efficient filtering.

        Time Complexity: O(n * m) where n is number of words and m is max word length.
        """
        # Store words by length for quick filtering
        self.words_by_length = defaultdict(list)
        print(f"Preprocessing words from {file}")
        file = settings.words_file.absolute() if file is None else file.absolute()
        # assert file.exists()

        with open(file, "r") as f:  # noqa: PTH123
            for word in f:
                word = word.strip().upper()  # noqa: PLW2901
                if min_length is None or len(word) >= min_length:
                    self.words_by_length[len(word)].append(word)

    def parse_pattern(self, pattern_str: str) -> List[set[str]]:
        """Convert pattern string to list of character sets."""
        return [self.char_sets[part.strip()] for part in pattern_str.split()]

    def find_matches(self, pattern_str: str) -> List[str]:
        """
        Find all words matching the given pattern.

        Time Complexity: O(n * m) where n is number of candidate words and m is pattern length
        """
        # Parse pattern into character sets
        pattern_sets = self.parse_pattern(pattern_str)
        pattern_length = len(pattern_sets)

        # Get words of the correct length
        candidate_words = self.words_by_length[pattern_length]
        matches = []

        # Check each word against the pattern
        for word in candidate_words:
            if all(letter in pattern_sets[i] for i, letter in enumerate(word)):
                matches.append(word)

        return matches


def main() -> None:
    """Main function for testing."""
    # Initialize matcher
    file = Path(r"../data/words_alpha.txt")
    matcher = PatternMatcher(file=file)

    # Example pattern
    pattern = "N-T G-M U-Z U-Z A-F"

    # Preprocess words (do this once)
    print("Preprocessing words...")
    start_time = time.time()
    matcher.preprocess_words(min_length=len(pattern.split()), file=file)
    print(f"Preprocessing completed in {time.time() - start_time:.2f} seconds")

    # Search for matches
    print(f"\nSearching for pattern: {pattern}")
    start_time = time.time()
    matches = matcher.find_matches(pattern)
    search_time = time.time() - start_time

    print(f"Found {len(matches)} matches in {search_time:.4f} seconds")
    print("Sample matches:", matches[:10])

    # Example showing PARIS would match:
    test_words = ["PARIS", "PIZZA"]
    pattern_sets = matcher.parse_pattern(pattern)
    for test_word in test_words:
        is_match = all(letter in pattern_sets[i] for i, letter in enumerate(test_word))
        print(f"\nExample: '{test_word}' matches pattern: {is_match}")
        if is_match:
            print(
                f"Because: P∈{pattern_sets[0]}, A∈{pattern_sets[1]}, "
                f"R∈{pattern_sets[2]}, I∈{pattern_sets[3]}, S∈{pattern_sets[4]}",
            )


if __name__ == "__main__":
    main()
