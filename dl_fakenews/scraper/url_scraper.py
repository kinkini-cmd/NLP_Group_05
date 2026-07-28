from newspaper import Article, Config


USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)


def extract_article(url):
    url = str(url).strip()

    if not url:
        return {
            "error": "URL is empty"
        }

    if not url.startswith(("http://", "https://")):
        return {
            "error": "Invalid URL. URL must start with http:// or https://"
        }

    try:
        config = Config()
        config.browser_user_agent = USER_AGENT
        config.request_timeout = 10

        article = Article(url, config=config)

        article.download()

        article.parse()

        title = article.title.strip()
        text = article.text.strip()

        if not title and not text:
            return {
                "error": "Could not extract article text. Try copying and pasting the full news text instead."
            }

        if not text:
            return {
                "error": "Article text not found. Try copying and pasting the full news text instead.",
                "title": title
            }

        return {

            "title":
                title,

            "text":
                text

        }


    except Exception as e:


        return {

            "error":
                str(e)

        }
