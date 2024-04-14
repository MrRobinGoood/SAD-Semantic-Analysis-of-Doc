import re

import pandas as pd
from pymorphy2 import MorphAnalyzer
import nltk
# nltk.download('stopwords', download_dir='resources/nltk_data')
from nltk.corpus import stopwords

nltk.data.path.append('resources/nltk_data')

stopwords_ru = stopwords.words("russian")
morph = MorphAnalyzer()


def clear_text(text: str):
    patterns = "[0-9!#$%&'()*+,./:;<=>?@[\]^_`{|}~—\"\-–•]+"
    text = re.sub(patterns, ' ', text)
    tokens = []
    for token in text.split():
        token = token.strip()
        token = morph.normal_forms(token)[0]
        if token not in stopwords_ru:
            tokens.append(token)
    return tokens


def clear_text_list(text: list):
    data = []
    for i in text:
        data.append(" ".join(clear_text(i)))
    return data


if __name__ == '__main__':
    df = pd.read_csv("dataset/sample.csv")
    print(clear_text_list(list(df['text'][:1])))
