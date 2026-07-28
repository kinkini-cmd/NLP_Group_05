
import pandas as pd
from dl_model.xlm_roberta.predict_xlmr import predict_news

real = pd.read_csv("datasets/raw/True.csv")

for i in range(5):
    article = real.iloc[i]["text"]
    print(i, predict_news(article))
