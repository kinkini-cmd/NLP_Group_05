import pandas as pd
import os

# Load datasets
true_df = pd.read_csv("data/True.csv")
fake_df = pd.read_csv("data/Fake.csv")

# Add labels: 1 = real, 0 = fake
true_df["label"] = 1
fake_df["label"] = 0

# Merge
df = pd.concat([true_df, fake_df], ignore_index=True)

# Combine title + text
df["content"] = df["title"] + " " + df["text"]

# Save merged dataset
os.makedirs("data", exist_ok=True)
df.to_csv("data/merged_dataset.csv", index=False)

print(f"Total samples: {len(df)}")
print(df["label"].value_counts())