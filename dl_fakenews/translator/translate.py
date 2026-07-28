from deep_translator import GoogleTranslator



def translate_to_english(text):

    try:

        result = GoogleTranslator(
            source="auto",
            target="en"
        ).translate(text)


        return result


    except Exception as e:

        print(
            "Translation error:",
            e
        )

        return text



if __name__ == "__main__":


    text = """
    ශ්‍රී ලංකාවේ රජය අද නව ප්‍රතිපත්තියක් හඳුන්වා දී ඇත
    """


    print(
        translate_to_english(text)
    )