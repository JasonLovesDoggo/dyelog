import logging
import string
from typing import ClassVar, List, Optional, Tuple

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from dyelog.settings import settings
from dyelog.utils import PatternMatcher
from ollama import AsyncClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
router = APIRouter()

MODEL = settings.ollama_model

matcher = PatternMatcher()


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
    """Enhanced model for word suggestions with confidence score."""

    id: int
    prompt: str
    confidence: float


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


async def score_words(words: List[str], context: str) -> List[Tuple[str, float]]:
    """
    Score words based on their relevance to the context using llama3.2.

    Returns list of (word, confidence) tuples.
    """
    if not words:
        return []

    prompt = f"""You are helping score words for someone with ALS to communicate.
Given the context: "{context}"

Please rate how appropriate and relevant each of these words is for the context:
{', '.join(words)}

For each word, return a single line with the format:
WORD:CONFIDENCE_SCORE

Where CONFIDENCE_SCORE is a number between 0 and 100 representing how appropriate
and relevant the word is for the given context.

Example output format:
PIZZA:85.5
PASTA:72.3

Only return the word:score pairs, one per line. Nothing else."""

    try:
        response = await client.chat(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            stream=False,
        )

        scored_words = []
        for line in response["message"]["content"].split("\n"):
            if ":" in line:
                word, score_str = line.strip().split(":")
                try:
                    score = float(score_str)
                    if score >= 30.0:  # Only include words with confidence >= 30%
                        scored_words.append((word.strip(), score))
                except ValueError:
                    continue

        return sorted(scored_words, key=lambda x: x[1], reverse=True)
    except Exception as e:
        logger.error(f"Error scoring words: {e}")
        return [(word, 60.0) for word in words]  # Fallback scoring


async def generate_words(
    context: str,
    letter_ranges: str,
    min_length: Optional[int] = None,
) -> List[Tuple[str, float]]:
    """Generate and score words matching the letter pattern using PatternMatcher and llama3.2."""
    try:
        # Initialize PatternMatcher

        # Convert letter ranges to pattern format
        pattern = letter_ranges.upper()

        # Get matching words
        matching_words = matcher.find_matches(pattern)
        print(f"Found {len(matching_words)} matching words ")
        print(f"{matching_words=}")
        if not matching_words:
            return []

        # Score and sort the matching words
        scored_words = await score_words(matching_words, context)

        return scored_words

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

    Now includes confidence scores for each word.
    """
    try:
        # Generate and score matching words
        scored_words = await generate_words(
            input.context,
            input.letter_ranges,
            min_length=len(input.letter_ranges.split()),
        )

        if not scored_words:
            return ChatResponse(prompt_options=[], sentences=[])

        # Create prompt options with confidence scores
        prompt_options = [
            PromptOption(id=i, prompt=word, confidence=score)
            for i, (word, score) in enumerate(scored_words)
        ]

        # Generate sentences using the highest confidence word
        sentences = await generate_sentences(scored_words[0][0], input.context)

        return ChatResponse(
            prompt_options=prompt_options,
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
