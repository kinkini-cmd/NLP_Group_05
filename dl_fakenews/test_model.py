import pandas as pd
from dl_model.xlm_roberta.predict_xlmr import predict_news

real = pd.read_csv("datasets/raw/True.csv")

article = real.iloc[0]["text"]

print(predict_news(article))
