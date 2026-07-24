import pandas as pd
from models.svm.predict_svm import predict_news

real = pd.read_csv("datasets/raw/True.csv")

article = real.iloc[0]["text"]

print(predict_news(article))
