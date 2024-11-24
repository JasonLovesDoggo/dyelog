def find_pattern(word: str) -> str:
    """
    Find the pattern that matches the given word.

    Args:
        word (str): The word to find a pattern for

    Returns:
        str: Space-separated pattern (e.g., "A-F G-M N-T U-Z U-Z")

    Example:
        >>> find_pattern("PARIS")
        "N-T A-F N-T G-M N-T"
    """
    sets = {
        "A-F": set("ABCDEF"),
        "G-M": set("GHIJKLM"),
        "N-T": set("NOPQRST"),
        "U-Z": set("UVWXYZ"),
    }

    if not word:
        return ""

    pattern_parts = []
    word = word.upper()

    for letter in word:
        found = False
        for set_name, char_set in sets.items():
            if letter in char_set:
                pattern_parts.append(set_name)
                found = True
                break
        if not found:
            raise ValueError(f"Letter '{letter}' not found in any character set")

    return " ".join(pattern_parts)


# # Example test cases
# def test_pattern_finder():
#     assert find_pattern("PARIS") == "N-T A-F N-T G-M N-T"
#     assert find_pattern("PIZZA") == "N-T G-M U-Z U-Z A-F"
#     assert find_pattern("HELLO") == "G-M A-F G-M G-M N-T"
#     assert find_pattern("") == ""
#     try:
#         find_pattern("123")
#         assert False, "Should raise ValueError for invalid characters"
#     except ValueError:
#         pass
#     print("All tests passed!")

if __name__ == "__main__":
    # test_pattern_finder()

    # Interactive testing
    while True:
        word = input("Enter a word (or press Enter to quit): ").strip()
        if not word:
            break
        try:
            pattern = find_pattern(word)
            print(f"Pattern for {word}: {pattern}")
        except ValueError as e:
            print(f"Error: {e}")
