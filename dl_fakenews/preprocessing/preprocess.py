# Import required libraries for text processing
import re                      # Used for removing URLs, numbers, patterns
import string                  # Used for removing punctuation
import nltk                    # Natural Language Toolkit for NLP tasks

from nltk.corpus import stopwords              # Common English words like "is", "the"
from nltk.stem import WordNetLemmatizer        # Reduces words to base form (e.g., running → run)

def load_stop_words():
    try:
        return set(stopwords.words('english'))
    except LookupError:
        return set()


# Load English stopwords into a set (faster lookup)
stop_words = load_stop_words()

# Create lemmatizer object
lemmatizer = WordNetLemmatizer()


# ---------------------------------------------
# Function 1: Clean text
# ---------------------------------------------
def clean_text(text):
    """
    This function cleans raw text by:
    - converting to lowercase
    - removing URLs
    - removing numbers
    - removing punctuation
    """

    # Convert text to string and lowercase it
    text = str(text).lower()

    # Remove URLs (http, https, www links)
    text = re.sub(r'http\S+|www\S+', '', text)

    # Remove numbers (0-9)
    text = re.sub(r'\d+', '', text)

    # Remove punctuation like !, ?, ., ,
    text = text.translate(str.maketrans('', '', string.punctuation))

    return text


# ---------------------------------------------
# Function 2: Remove stopwords
# ---------------------------------------------
def remove_stopwords(text):
    """
    This function removes common English words
    that do not add meaning (e.g., is, the, and)
    """

    words = text.split()   # Split sentence into words

    # Keep only words not in stopword list
    words = [word for word in words if word not in stop_words]

    return " ".join(words)


# ---------------------------------------------
# Function 3: Lemmatization
# ---------------------------------------------
def lemmatize_text(text):
    """
    Converts words into their base form:
    Example:
    - running → run
    - cars → car
    """

    words = text.split()

    try:
        # Convert each word to its base form
        words = [lemmatizer.lemmatize(word) for word in words]
    except LookupError:
        return text

    return " ".join(words)


# ---------------------------------------------
# Function 4: Full preprocessing pipeline
# ---------------------------------------------
def preprocess_text(text):
    """
    This is the main function that:
    1. Cleans text
    2. Removes stopwords
    3. Lemmatizes words
    """

    text = clean_text(text)          # Step 1: Clean raw text
    text = remove_stopwords(text)    # Step 2: Remove useless words
    text = lemmatize_text(text)      # Step 3: Normalize words

    return text


# ---------------------------------------------
# Test example (to check if code works)
# ---------------------------------------------
if __name__ == "__main__":

    sample = "Breaking News!!! Government announces new policy at https://example.com"

    print("Original Text:")
    print(sample)

    print("\nProcessed Text:")
    print(preprocess_text(sample))
