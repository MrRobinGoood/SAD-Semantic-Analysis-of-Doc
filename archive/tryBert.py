import torch
from transformers import BertForSequenceClassification, BertTokenizerFast
import pandas as pd

# Загрузка токенизатора
tokenizer = BertTokenizerFast.from_pretrained('rubert-base-cased')

# Загрузка модели
model = BertForSequenceClassification.from_pretrained("my_model")
df = pd.read_csv('../dataset/test.csv')
# Пример предсказания
df['class'] = df['class'].map({'act': 0, 'application': 1,'arrangement': 2,'bill': 3,'contract': 4,'contract offer': 5,'determination': 6,'invoice': 7,'order': 8,'proxy': 9,'statute': 10})

loss = 0

for index, row in df.iterrows():
    text = row['text']
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=1)
    if predictions.item() != row['class']:
        print(row)
        loss += 1

print(loss)