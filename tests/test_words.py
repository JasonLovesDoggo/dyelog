# type: ignore

import os
import tempfile

import pytest

from dyelog.utils import PatternMatcher


@pytest.fixture
def sample_words_file():
    """Create a temporary file with test words"""
    words = [
        "PARIS",
        "HELLO",
        "WORLD",
        "TESTS",
        "HAPPY",
        "BRAIN",
        "TRAIN",
        "ZEBRA",
        "SHORT",
        "A",  # Too short
        "TOOLONG",  # Too long
    ]

    with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
        for word in words:
            f.write(word + "\n")
        temp_file_name = f.name

    yield temp_file_name

    # Cleanup
    os.unlink(temp_file_name)


@pytest.fixture
def matcher(sample_words_file):
    """Create and initialize a PatternMatcher instance"""
    matcher = PatternMatcher()
    matcher.preprocess_words(sample_words_file)
    return matcher


def test_initialization():
    """Test PatternMatcher initialization"""
    matcher = PatternMatcher()
    assert isinstance(matcher.char_sets, dict)
    assert "A-F" in matcher.char_sets
    assert "G-M" in matcher.char_sets
    assert "N-T" in matcher.char_sets
    assert "U-Z" in matcher.char_sets


def test_char_sets_content():
    """Test the content of character sets"""
    matcher = PatternMatcher()
    assert "A" in matcher.char_sets["A-F"]
    assert "F" in matcher.char_sets["A-F"]
    assert "G" in matcher.char_sets["G-M"]
    assert "M" in matcher.char_sets["G-M"]
    assert "N" in matcher.char_sets["N-T"]
    assert "T" in matcher.char_sets["N-T"]
    assert "U" in matcher.char_sets["U-Z"]
    assert "Z" in matcher.char_sets["U-Z"]


def test_parse_pattern():
    """Test pattern parsing"""
    matcher = PatternMatcher()
    pattern = "A-F G-M N-T U-Z U-Z"
    parsed = matcher.parse_pattern(pattern)

    assert len(parsed) == 5
    assert all(isinstance(char_set, set) for char_set in parsed)
    assert "A" in parsed[0]
    assert "M" in parsed[1]
    assert "T" in parsed[2]
    assert "Z" in parsed[3]
    assert parsed[3] == parsed[4]  # Last two sets should be identical


def test_find_matches_paris(matcher):
    """Test finding matches with pattern that should match 'PARIS'"""
    pattern = "N-T A-F N-T G-M N-T"  # Pattern that should match PARIS
    matches = matcher.find_matches(pattern)
    assert "PARIS" in matches


def test_find_matches_length_filter(matcher):
    """Test that words of wrong length are filtered out"""
    pattern = "A-F G-M N-T U-Z U-Z"  # 5-letter pattern
    matches = matcher.find_matches(pattern)

    # Check that all matches are exactly 5 letters
    assert all(len(word) == 5 for word in matches)
    assert "A" not in matches  # Too short
    assert "TOOLONG" not in matches  # Too long


def test_find_matches_case_sensitivity(sample_words_file):
    """Test case sensitivity handling"""
    matcher = PatternMatcher()
    matcher.preprocess_words(sample_words_file)

    pattern = "N-T A-F N-T G-M N-T"  # Pattern for PARIS
    matches = matcher.find_matches(pattern)

    # The matcher should work with uppercase input
    assert "PARIS" in matches


@pytest.mark.parametrize(
    "pattern,word,should_match",
    [
        ("N-T A-F N-T G-M N-T", "PARIS", True),  # Valid match
        ("A-F G-M N-T U-Z U-Z", "PARIS", False),  # Wrong pattern
        ("N-T A-F N-T G-M N-T", "HELLO", False),  # Non-matching word
        ("N-T N-T A-F G-M N-T", "TRAIN", True),  # Another valid match
    ],
)
def test_specific_word_patterns(matcher, pattern, word, should_match):
    """Test specific word-pattern combinations"""
    matches = matcher.find_matches(pattern)
    assert (word in matches) == should_match


def test_empty_file():
    """Test handling of empty input file"""
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
        temp_file_name = f.name

    matcher = PatternMatcher()
    matcher.preprocess_words(temp_file_name)

    pattern = "A-F G-M N-T U-Z U-Z"
    matches = matcher.find_matches(pattern)

    assert len(matches) == 0

    os.unlink(temp_file_name)


def test_all_char_sets_disjoint():
    """Test that character sets don't overlap"""
    matcher = PatternMatcher()
    all_chars = set()

    for char_set in matcher.char_sets.values():
        # Check that this set doesn't overlap with previously seen chars
        assert not (char_set & all_chars)
        all_chars.update(char_set)


def test_preprocessing_idempotent(sample_words_file):
    """Test that preprocessing multiple times doesn't change results"""
    matcher = PatternMatcher()

    # First preprocessing
    matcher.preprocess_words(sample_words_file)
    first_words = dict(matcher.words_by_length)

    # Second preprocessing
    matcher.preprocess_words(sample_words_file)
    second_words = dict(matcher.words_by_length)

    assert first_words == second_words


def test_performance_large_input(tmp_path):
    """Test performance with larger input"""
    # Create a file with 1000 random 5-letter words
    large_file = tmp_path / "large_wordlist.txt"
    with open(large_file, "w") as f:
        for i in range(1000):
            f.write("PARIS\n")  # Using PARIS as a dummy word

    matcher = PatternMatcher()
    matcher.preprocess_words(str(large_file))

    pattern = "N-T A-F N-T G-M N-T"
    matches = matcher.find_matches(pattern)

    assert len(matches) > 0
    assert "PARIS" in matches


if __name__ == "__main__":
    pytest.main(["-v"])
