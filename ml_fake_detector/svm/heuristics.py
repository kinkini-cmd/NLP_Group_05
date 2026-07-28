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

SINHALA_REAL_WORDS = [
    "ජනාධිපති",
    "අගමැති",
    "පාර්ලිමේන්තුව",
    "පක්ෂ",
    "අමාත්‍ය",
    "අමාත්‍යාංශය",
    "රජය",
    "පොලිසිය",
    "රෝහල",
    "අධිකරණය",
    "වාර්තා",
    "ප්‍රකාශ",
    "හමුවීමට",
    "ඉල්ලයි",
]

SINHALA_FAKE_WORDS = [
    "අදහාගත නොහැකි",
    "රහස",
    "හෙළිවෙයි",
    "විශ්මිත",
    "ක්ෂණිකව",
    "වයිරල්",
]

LOCAL_INCIDENT_PATTERNS = [
    ("police", "hospital"),
    ("police", "accident"),
    ("accident", "hospital"),
    ("collide", "hospital"),
    ("dead", "critical"),
    ("injured", "hospital"),
]


def contains_sinhala(text):
    return bool(re.search(r"[\u0D80-\u0DFF]", text))


def heuristic_score(text):
    text_lower = text.lower()
    score = 0
    is_sinhala = contains_sinhala(text)

    for word in FAKE_WORDS:
        if word in text_lower:
            score -= 1

    for word in REAL_WORDS:
        if word in text_lower:
            score += 1

    for first, second in LOCAL_INCIDENT_PATTERNS:
        if first in text_lower and second in text_lower:
            score += 1

    if is_sinhala:
        for word in SINHALA_FAKE_WORDS:
            if word in text:
                score -= 1

        for word in SINHALA_REAL_WORDS:
            if word in text:
                score += 1

    if text.count("!") >= 3:
        score -= 1

    if len(re.findall(r"[A-Z]{4,}", text)) >= 3:
        score -= 1

    if len(text.split()) < 20 and not is_sinhala:
        score -= 0.15

    return score
