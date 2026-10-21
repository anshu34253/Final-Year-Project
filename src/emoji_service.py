"""
src/emoji_service.py
--------------------
Transcript-Aware Contextual Emoji Enrichment Service.
Inspects spoken dialogue transcripts and word timestamps to append
contextually relevant emojis to caption lines, drawing on the pycaps
semantic tagging and emoji effect architecture.
"""

import re
import random
from typing import List, Dict, Any, Optional, Tuple

from src.utf8_utils import setup_utf8_environment
setup_utf8_environment()

# Contextual Semantic Emoji Map (keywords -> list of relevant emojis)
SEMANTIC_EMOJI_LEXICON = {
    # Amazement & Shock
    "amazed": (
        ["wow", "amazing", "incredible", "unbelievable", "insane", "omg", "shocking", "woah", "astonishing", "speechless"],
        ["😲", "😮", "🤯"]
    ),
    # Mind-blown / Genius
    "mindblown": (
        ["mindblown", "genius", "brain", "unreal", "magic", "smart", "brilliant", "iq"],
        ["🤯", "🧠", "✨"]
    ),
    # Fire / Energy / Hype
    "fire": (
        ["fire", "lit", "hot", "flame", "crazy", "wild", "beast", "epic", "huge", "monster", "hyped", "explosive"],
        ["🔥", "⚡", "💥"]
    ),
    # Money / Wealth / Value
    "money": (
        ["money", "cash", "dollar", "dollars", "rich", "wealth", "expensive", "profit", "crypto", "billion", "million", "cost", "price"],
        ["💰", "💵", "🤑"]
    ),
    # Victory / Championship / Success
    "win": (
        ["win", "winner", "winning", "champion", "championship", "victory", "first", "goat", "best", "legend", "trophy", "king"],
        ["🏆", "🥇", "👑"]
    ),
    # Speed / Rocket / Launch
    "speed": (
        ["rocket", "launch", "fast", "speed", "boost", "skyrocket", "explode", "growth", "quick", "faster", "rush"],
        ["🚀", "🏎️", "⚡"]
    ),
    # Emotion / Love / Passion
    "love": (
        ["love", "heart", "passion", "favorite", "beloved", "beautiful", "gorgeous", "sweet"],
        ["❤️", "💖", "😍"]
    ),
    # Humor / Laugh
    "humor": (
        ["funny", "joke", "hilarious", "laugh", "laughing", "lol", "lmao", "funniest"],
        ["😂", "🤣"]
    ),
    # Warning / Caution / Alert
    "warning": (
        ["warning", "warn", "danger", "dangerous", "careful", "beware", "stop", "mistake", "never", "risk", "trap"],
        ["⚠️", "🚨", "🛑"]
    ),
    # Verification / Correct / Agreement
    "truth": (
        ["yes", "true", "correct", "perfect", "proven", "solution", "agree", "fact", "facts", "definitely", "absolutely"],
        ["✅", "💯", "🎯"]
    ),
    # Idea / Secret / Strategy
    "idea": (
        ["idea", "tip", "strategy", "secret", "hack", "advice", "think", "thought", "knowledge", "learn", "how"],
        ["💡", "🔑", "📌"]
    ),
    # Music / Audio
    "music": (
        ["music", "song", "beat", "track", "sing", "sound", "tune", "audio"],
        ["🎵", "🎶", "🎧"]
    ),
    # Eyes / Look / Watch
    "vision": (
        ["look", "see", "watch", "eyes", "notice", "reveal", "check", "observe"],
        ["👀", "🔍"]
    ),
    # Strength / Power / Gym
    "strength": (
        ["strong", "strength", "power", "heavy", "lift", "muscle", "workout", "fitness", "body", "punch"],
        ["💪", "🦾"]
    )
}

# Pre-compile fast word lookup table: lowercase_word -> (category, list_of_emojis)
_WORD_LOOKUP: Dict[str, Tuple[str, List[str]]] = {}
for category, (wordlist, emoji_choices) in SEMANTIC_EMOJI_LEXICON.items():
    for word in wordlist:
        _WORD_LOOKUP[word.lower()] = (category, emoji_choices)


def detect_emoji_for_text(text: str, last_category: Optional[str] = None) -> Optional[Tuple[str, str]]:
    """
    Scans text for semantic trigger words.
    Returns (emoji_character, category_name) or None if no semantic match found.
    Avoids returning an emoji from the same category as last_category to prevent repetition.
    """
    if not text:
        return None

    # Clean punctuation and tokenize words
    words = re.findall(r"\b[a-zA-Z0-9']+\b", text.lower())
    if not words:
        return None

    matched_categories = []
    for w in words:
        if w in _WORD_LOOKUP:
            cat, emojis = _WORD_LOOKUP[w]
            if cat != last_category:
                matched_categories.append((cat, emojis))

    if not matched_categories:
        # If all matches were the same category, pick one if available
        for w in words:
            if w in _WORD_LOOKUP:
                cat, emojis = _WORD_LOOKUP[w]
                matched_categories.append((cat, emojis))
                break

    if not matched_categories:
        return None

    # Pick the most prominent match
    chosen_cat, chosen_emojis = matched_categories[0]
    return random.choice(chosen_emojis), chosen_cat


def enhance_segments_with_emojis(
    speech_segments: List[Dict[str, Any]],
    enabled: bool = True
) -> List[Dict[str, Any]]:
    """
    Enriches subtitle speech segments with contextual emojis if enabled.
    Returns a new list of segments with enhanced text.
    """
    if not enabled or not speech_segments:
        return speech_segments

    enhanced_segments = []
    last_cat = None

    for seg in speech_segments:
        new_seg = dict(seg)
        raw_text = seg.get("text", "").strip()

        # Check if segment already has emojis
        match = detect_emoji_for_text(raw_text, last_category=last_cat)
        if match:
            emoji_char, cat = match
            last_cat = cat
            # Append emoji naturally to the end of the phrase
            if not raw_text.endswith(emoji_char):
                new_seg["text"] = f"{raw_text} {emoji_char}"
        else:
            last_cat = None

        enhanced_segments.append(new_seg)

    return enhanced_segments
