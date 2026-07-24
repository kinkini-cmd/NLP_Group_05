import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


nltk.download("stopwords")
nltk.download("wordnet")

df = pd.read_csv("data/merged_dataset.csv")
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = str(text).lower()                          # Lowercase
    text = re.sub(r"http\S+|www\S+", "", text)       # Remove URLs
    text = re.sub(r"[^a-z\s]", "", text)             # Remove special chars/numbers
    tokens = text.split()
    tokens = [t for t in tokens if t not in stop_words]  # Remove stopwords
    tokens = [lemmatizer.lemmatize(t) for t in tokens]   # Lemmatization
    return " ".join(tokens)

df["clean_text"] = df["content"].apply(clean_text)
df.dropna(subset=["clean_text"], inplace=True)
df.to_csv("data/preprocessed_dataset.csv", index=False)

print("Preprocessing complete.")
print(df[["content", "clean_text", "label"]].head(3))
