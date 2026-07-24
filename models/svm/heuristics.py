import re


FAKE_WORDS = [
    "shocking",
    "secret",
    "exposed",
    "you won't believe",
    "miracle",
    "breaking",
    "viral",
    "must see",
    "truth revealed",
]

REAL_WORDS = [
    "according to",
    "official statement",
    "reported by",
    "data shows",
    "court",
    "ministry",
    "police",
    "police said",
    "hospital",
    "critical condition",
    "accident",
    "accident occurred",
    "admitted to hospital",
    "investigation",
    "collision",
    "collide",
    "head-on",
    "injured",
]

LOCAL_INCIDENT_PATTERNS = [
    ("police", "hospital"),
    ("police", "accident"),
    ("accident", "hospital"),
    ("collide", "hospital"),
    ("dead", "critical"),
    ("injured", "hospital"),
]


def heuristic_score(text):
    text_lower = text.lower()
    score = 0

    for word in FAKE_WORDS:
        if word in text_lower:
            score -= 1

    for word in REAL_WORDS:
        if word in text_lower:
            score += 1

    for first, second in LOCAL_INCIDENT_PATTERNS:
        if first in text_lower and second in text_lower:
            score += 1

    if text.count("!") >= 3:
        score -= 1

    if len(re.findall(r"[A-Z]{4,}", text)) >= 3:
        score -= 1

    if len(text.split()) < 40:
        score -= 0.5

    return score
