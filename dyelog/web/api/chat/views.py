import logging
import string
from typing import ClassVar, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from dyelog.settings import settings
from ollama import AsyncClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
MODEL = "llama3.2:3b-instruct-fp16"
router = APIRouter()


# Models
class ChatInput(BaseModel):
    """Input model for chat requests."""

    letter_ranges: str = Field(  # type: ignore
        ...,
        description="Letter ranges in format 'A-C D-F G-I'. "
        "Each range represents a position in the word.",
        example="N-T G-M U-Z U-Z A-F",
    )
    context: str = Field(  # type: ignore
        ...,
        description="The conversational context or question being answered",
        example="What would you like to eat?",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "letter_ranges": "N-T G-M U-Z U-Z A-F",
                "context": "What would you like to eat?",
            },
        }


class PromptOption(BaseModel):
    """Model for word suggestions."""

    id: int
    prompt: str


class ChatResponse(BaseModel):
    """Response model containing word options and generated sentences."""

    prompt_options: List[PromptOption]
    sentences: List[str]

    class Config:
        json_schema_extra: ClassVar = {
            "example": {
                "prompt_options": [
                    {"id": 0, "prompt": "PIZZA"},
                    {"id": 1, "prompt": "PASTA"},
                    {"id": 2, "prompt": "SALAD"},
                ],
                "sentences": [
                    "I would like to eat pizza for dinner.",
                    "Can we order a pizza tonight?",
                    "Pizza sounds perfect right now.",
                    "I'm craving a hot pizza with extra cheese.",
                ],
            },
        }


class WordSelector(BaseModel):
    """Model for word selection."""

    word: str = Field(..., example="PIZZA")  # type: ignore
    context: str = Field(..., example="What would you like to eat?")  # type: ignore


# Initialize FastAPI and Ollama client

client = AsyncClient(host=settings.ollama_host)


def parse_letter_ranges(ranges_str: str) -> List[set[str]]:
    """
    Parse letter ranges string into sets of allowed letters.

    Args:
        ranges_str (str): Space-separated letter ranges (e.g., "A-C D-F")

    Returns:
        List[set[str]]: List of sets containing allowed letters for each position

    Example:
        >>> parse_letter_ranges("A-C D-F")
        [{A,B,C}, {D,E,F}]
    """
    letter_sets = []
    ranges = ranges_str.upper().split()

    for range_str in ranges:
        if "-" not in range_str:
            letter_sets.append({range_str})
            continue

        start, end = range_str.split("-")
        start_idx = string.ascii_uppercase.index(start)
        end_idx = string.ascii_uppercase.index(end)
        letters = set(string.ascii_uppercase[start_idx : end_idx + 1])
        letter_sets.append(letters)

    return letter_sets


def validate_word(word: str, letter_sets: List[set[str]]) -> bool:
    """
    Validate if a word matches the letter pattern requirements.

    Args:
        word (str): Word to validate
        letter_sets (List[set[str]]): List of sets containing allowed letters

    Returns:
        bool: True if word matches pattern, False otherwise
    """
    if not word or len(word) < len(letter_sets):
        return False

    return all(
        letter.upper() in letter_set for letter, letter_set in zip(word, letter_sets)
    )


async def generate_words(
    context: str,
    letter_ranges: str,
    min_length: Optional[int] = None,
) -> List[str]:
    """Generate words matching the letter pattern using Ollama."""
    letter_sets = parse_letter_ranges(letter_ranges)
    min_length = min_length or len(letter_sets)

    pattern_desc = " ".join(f"[{''.join(sorted(s))}]" for s in letter_sets)
    prompt = f"""You are helping generate words for someone with ALS to communicate.
Generate 1-4 common English words that match this pattern: {pattern_desc}.
The context surrounding this response is: {context}

Requirements:
1. Words must be at least {min_length} letters long
2. Each letter must match its position's allowed letters
3. Words must be common and useful for everyday communication
4. Words should be appropriate for the context

__Example__ patterns and valid words:
Pattern: [NOPQRST] [GHIJKLM] [UVWXYZ] [UVWXYZ] [ABCDEF]
Valid: PIZZA

Pattern: [ABC] [DEF] [GHI] [JKL]
Valid: BEAK

Pattern: [WXYZ] [ABCD] [NOPQ] [TUVW]
Valid: WANT

Do not return responses for the example patterns.

Return only valid words, one per line. If unsure, return nothing."""

    try:
        response = await client.chat(
            model=MODEL,  # Using llama3.2 instruct model
            messages=[{"role": "user", "content": prompt}],
            stream=False,
        )
        print(response["message"]["content"])

        return [
            word.strip().upper()
            for word in response["message"]["content"].split("\n")
            if word.strip() and validate_word(word.strip(), letter_sets)
        ]

    except Exception as e:
        logger.error(f"Error generating words: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate words")


async def generate_sentences(word: str, context: str) -> List[str]:
    """Generate contextually appropriate sentences using the selected word."""
    prompt = f"""You are helping generate natural sentences for someone with ALS to communicate.

Generate 1-4 conversational sentences that:
1. Use the word "{word}"
2. Are appropriate responses to: "{context}"
3. Vary in structure and tone
4. Are clear and direct
5. Sound natural in conversation

Examples:
Context: "What would you like to eat?"
Word: "PIZZA"
Sentences:
- I would like to eat pizza.
- Can we order pizza tonight?
- Pizza sounds perfect right now.

Context: "How are you feeling?"
Word: "TIRED"
Sentences:
- I'm feeling tired today.
- Tired, I need some rest.
- Could we talk later? I'm tired.

Return only the sentences, one per line. If unsure, return nothing."""

    try:
        response = await client.chat(
            model=MODEL,  # Using llama3.2 instruct model
            messages=[{"role": "user", "content": prompt}],
            stream=False,
        )

        return [
            sent.strip()
            for sent in response["message"]["content"].split("\n")
            if sent.strip() and not sent.startswith("-")  # Filter out any bullet points
        ]

    except Exception as e:
        logger.error(f"Error generating sentences: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate sentences")


@router.post("/predict", response_model=ChatResponse, tags=["prediction"])
async def predict(input: ChatInput) -> ChatResponse:
    """
    Generate word predictions and example sentences based on letter ranges and context.

    Args:
        input (ChatInput): Contains letter_ranges and context
            - letter_ranges: String of letter ranges (e.g., "N-T G-M U-Z U-Z A-F")
            - context: Conversational context or question

    Returns:
        ChatResponse: Contains word options and example sentences
            - prompt_options: List of suggested words matching the pattern
            - sentences: List of example sentences using the first suggested word

    Examples:
        Request:
        ```json
        {
            "letter_ranges": "N-T G-M U-Z U-Z A-F",
            "context": "What would you like to eat?"
        }
        ```

        Response:
        ```json
        {
            "prompt_options": [
                {"id": 0, "prompt": "PIZZA"},
                {"id": 1, "prompt": "PASTA"}
            ],
            "sentences": [
                "I would like to eat pizza.",
                "Can we order pizza tonight?",
                "Pizza sounds perfect right now."
            ]
        }
        ```

    Raises:
        HTTPException(404): If no matching words are found
        HTTPException(500): If word or sentence generation fails
    """
    try:
        # Generate matching words
        words = await generate_words(
            input.context,
            input.letter_ranges,
            min_length=len(input.letter_ranges),
        )
        if not words:
            return ChatResponse(prompt_options=[], sentences=[])

        # Generate sentences using the first word
        sentences = await generate_sentences(words[0], input.context)

        return ChatResponse(
            prompt_options=[
                PromptOption(id=i, prompt=word) for i, word in enumerate(words)
            ],
            sentences=sentences,
        )

    except Exception as e:
        logger.error(f"Error in predict endpoint: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to generate predictions",
        )


@router.post("/sentences", tags=["sentences"])
async def get_sentences(selector: WordSelector) -> List[str]:
    """Generate new sentences for a selected word and context."""
    try:
        return await generate_sentences(selector.word, selector.context)
    except Exception as e:
        logger.error(f"Error generating sentences: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to generate sentences",
        )
