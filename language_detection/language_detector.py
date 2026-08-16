import re


def detect_language(text):

    if not text:
        return "unknown"


    # Sinhala Unicode block
    sinhala = re.findall(
        r'[\u0D80-\u0DFF]',
        text
    )

    if len(sinhala) >= 3:
        return "si"


    # English detection
    english = re.findall(
        r'[A-Za-z]',
        text
    )

    if len(english) >= 3:
        return "en"


    return "unknown"



if __name__ == "__main__":

    print("English:")
    print(
        detect_language(
            "Government announces new policy today"
        )
    )


    print("Sinhala:")
    print(
        detect_language(
            "ශ්‍රී ලංකාවේ රජය අද නව ප්‍රතිපත්තියක් හඳුන්වා දී ඇත"
        )
    )