from __future__ import annotations

import string
from typing import Dict, List

from fastapi import APIRouter, HTTPException
from fastapi.logger import logger
from pydantic import BaseModel

from dyelog.settings import settings
from ollama import AsyncClient


# Models
class ChatInput(BaseModel):
    """Chat input model."""

    letter_ranges: str  # e.g. "N-T G-M U-Z U-Z A-F"
    context: str


class PromptOption(BaseModel):
    """Prompt option model."""

    id: int
    prompt: str


class ChatResponse(BaseModel):
    """Chat response model."""

    prompt_options: List[PromptOption]
    sentences: List[str]


class WordSelection(BaseModel):
    """Selected word model."""

    selected_word: str


# Global session state
session_words: List[str] = []
session_context: List[str] = []

# Initialize Ollama client
client = AsyncClient(host=settings.ollama_host)

router = APIRouter()


def parse_letter_ranges(ranges_str: str) -> List[set[str]]:
    """Convert letter ranges string to sets of allowed letters."""
    letter_sets = []
    ranges = ranges_str.split()

    for range_str in ranges:
        if "-" not in range_str:
            letter_sets.append({range_str.upper()})
            continue

        start, end = range_str.split("-")
        letters = set(
            string.ascii_uppercase[
                string.ascii_uppercase.index(
                    start.upper(),
                ) : string.ascii_uppercase.index(end.upper())
                + 1
            ],
        )
        letter_sets.append(letters)

    return letter_sets


def word_matches_pattern(word: str, letter_sets: List[set[str]]) -> bool:
    """Check if a word matches the letter pattern."""
    if len(word) < len(letter_sets):
        return False

    return all(
        letter.upper() in letter_set for letter, letter_set in zip(word, letter_sets)
    )


async def generate_word_options(letter_ranges: str) -> List[str]:
    """Generate possible words matching the letter patterns using LLM."""
    letter_sets = parse_letter_ranges(letter_ranges)
    pattern_description = " ".join(
        f"[{''.join(sorted(letter_set))}]" for letter_set in letter_sets
    )
    logger.debug(f"pattern_description={pattern_description}")

    prompt = f"""
Generate 4 common English words that use the letters from the following pattern: {letter_ranges}
    Each position can only use letters from its corresponding bracket.
    Remember to ensure that the word is at least {len(letter_sets)} letters long.
    Return only the words, one per line, nothing else.
For example, if the pattern is N-T G-M U-Z U-Z A-F then,
the words could be: Pizza .
"""
    logger.debug(f"prompt={prompt}")
    message = {"role": "user", "content": prompt}

    try:
        response = await client.chat(
            model="llama3.2",
            messages=[message],
            stream=False,
        )  # Get the full response at once
        response_text = response["message"]["content"]
        print(f"response_text={response_text}")
        words = [
            word.strip().upper()
            for word in response_text.split("\n")
            if word.strip() and word_matches_pattern(word.strip(), letter_sets)
        ]
        print(f"{words=}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating words: {e!s}")

    return words[:4]


async def generate_sentences(word: str, context: str) -> List[str]:
    """Generate sentences incorporating the word and context."""
    prompt = f"""Generate 4 natural, grammatically correct sentences that:
    1. Use the word "{word}"
    2. Relate to the context: "{context}"
    3. Are appropriate for someone trying to communicate their needs/thoughts
    4. Are clear and concise
    Return only the sentences, one per line, nothing else."""

    message = {"role": "user", "content": prompt}
    sentences = []
    combined_response = ""

    try:
        async for part in await client.chat(
            model="llama3.2",
            messages=[message],
            stream=True,
        ):
            combined_response += part["message"]["content"]
            current_sentences = [
                sent.strip() for sent in combined_response.split("\n") if sent.strip()
            ]
            sentences = current_sentences
            if len(sentences) >= 4:
                break
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating sentences: {e!s}",
        )

    return sentences[:4]


@router.post("/predict", response_model=ChatResponse)
async def predict_words(incoming_message: ChatInput) -> ChatResponse:
    """Predict words based on letter ranges and generate corresponding sentences."""
    try:
        words = await generate_word_options(incoming_message.letter_ranges)
        logger.info(f"Generated word options: {words}")

        if not words:
            raise HTTPException(status_code=404, detail="No matching words found")

        # Generate sentences for the first word prediction
        sentences = await generate_sentences(words[0], incoming_message.context)

        return ChatResponse(
            prompt_options=[
                PromptOption(id=i, prompt=word) for i, word in enumerate(words)
            ],
            sentences=sentences,
        )
    except Exception as e:
        print(f"{e=}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/select-word")
async def select_word(selection: WordSelection) -> Dict[str, List[str] | str]:
    """Add selected word to session context."""
    session_words.append(selection.selected_word)
    return {"status": "success", "session_words": session_words}


@router.get("/session")
async def get_session() -> Dict[str, List[str]]:
    """Get current session state."""
    return {
        "words": session_words,
        "context": session_context,
    }


@router.post("/clear-session")
async def clear_session() -> Dict[str, str]:
    """Clear session state."""
    session_words.clear()
    session_context.clear()
    return {"status": "success"}
