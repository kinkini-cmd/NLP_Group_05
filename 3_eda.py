import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter


df = pd.read_csv("data/preprocessed_dataset.csv")

# 1. Class Distribution
df["label"].value_counts().plot(kind="bar", color=["tomato", "steelblue"])
plt.title("Class Distribution (0=Fake, 1=Real)")
plt.xticks(rotation=0)
plt.savefig("eda_class_distribution.png")
plt.show()

# 2. Text Length Distribution
df["text_length"] = df["clean_text"].apply(lambda x: len(str(x).split()))
df.groupby("label")["text_length"].hist(alpha=0.6, bins=50)
plt.title("Word Count Distribution")
plt.savefig("eda_word_count.png")
plt.show()

# 3. Word Cloud - Fake News
fake_text = " ".join(df[df["label"] == 0]["clean_text"].dropna().astype(str))
wc = WordCloud(width=800, height=400, background_color="white").generate(fake_text)
plt.imshow(wc); plt.axis("off"); plt.title("Fake News Word Cloud")
plt.savefig("eda_wordcloud_fake.png"); plt.show()

# 4. Top 20 words
all_words = " ".join(df["clean_text"].dropna().astype(str)).split()
freq = Counter(all_words).most_common(20)
words, counts = zip(*freq)
plt.barh(words, counts, color="coral")
plt.title("Top 20 Most Frequent Words")
plt.savefig("eda_top_words.png"); plt.show()
