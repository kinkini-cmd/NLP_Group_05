from ml_fake_detector.svm.predict_svm import predict_news


SAMPLES = [
    {
        "expected": "REAL",
        "source": "AP",
        "url": "https://www.ap.org/news-highlights/",
        "text": (
            "The ancient Buddhist site of Sarnath has been added to the "
            "World Heritage list, with AP photos showing novice monks "
            "walking in procession in Sarnath, India."
        ),
    },
    {
        "expected": "REAL",
        "source": "AP",
        "url": "https://www.ap.org/news-highlights/",
        "text": (
            "The annual mud festival opened in South Korea coastal city of "
            "Boryeong, where visitors played with mud near the beach during "
            "the festival."
        ),
    },
    {
        "expected": "REAL",
        "source": "AP",
        "url": "https://www.ap.org/news-highlights/",
        "text": (
            "AP reporters uncovered details about a Maine ICE officer who "
            "fatally shot Colombian national Johan Sebastian Duran Guerrero "
            "near his home in Biddeford, Maine."
        ),
    },
    {
        "expected": "REAL",
        "source": "AP",
        "url": "https://www.ap.org/news-highlights/",
        "text": (
            "AP World Cup coverage reported Spain won the World Cup final "
            "against Argentina in East Rutherford, New Jersey, with Ferran "
            "Torres celebrating with the trophy."
        ),
    },
    {
        "expected": "REAL",
        "source": "AP",
        "url": "https://www.ap.org/news-highlights/",
        "text": (
            "AP Bangkok team reported a fatal Bangkok music bar fire after "
            "a local contact alerted photographer Sakchai Lalit and "
            "reporters mobilized coverage from the scene."
        ),
    },
    {
        "expected": "FAKE",
        "source": "PolitiFact Pants on Fire",
        "url": None,
        "text": (
            "New York Democrats are communists for trying to force New "
            "Yorkers to conserve energy."
        ),
    },
    {
        "expected": "FAKE",
        "source": "PolitiFact Pants on Fire",
        "url": None,
        "text": (
            "Michigan U.S. Senate candidate Mike Rogers took a 14 million "
            "dollar payout as a pharma lobbyist."
        ),
    },
    {
        "expected": "FAKE",
        "source": "PolitiFact Pants on Fire",
        "url": None,
        "text": "Michelle Obama is a man.",
    },
    {
        "expected": "FAKE",
        "source": "PolitiFact Pants on Fire",
        "url": None,
        "text": (
            "The ongoing ballot counting in California means they are "
            "cheating on the election."
        ),
    },
    {
        "expected": "FAKE",
        "source": "PolitiFact Pants on Fire",
        "url": None,
        "text": (
            "This image shows a person stuffing ballots into a California "
            "ballot box."
        ),
    },
]


def run_evaluation():
    correct = 0

    print("id\texpected\tpredicted\tconfidence\tcorrect\tsource")

    for index, sample in enumerate(SAMPLES, start=1):
        prediction, confidence = predict_news(sample["text"], url=sample["url"])
        is_correct = prediction == sample["expected"]
        correct += int(is_correct)

        print(
            f"{index}\t{sample['expected']}\t{prediction}\t"
            f"{confidence}\t{is_correct}\t{sample['source']}"
        )

    total = len(SAMPLES)
    accuracy = (correct / total) * 100
    print(f"\nAccuracy: {correct}/{total} = {accuracy:.2f}%")


if __name__ == "__main__":
    run_evaluation()
