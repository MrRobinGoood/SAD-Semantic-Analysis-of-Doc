from joblib import load
from prepare_text import clear_text_list
import pandas as pd

model = load('model/model1.joblib')
vectorized = load('model/vector1.joblib')

def predict_class(text):
    cleaned_text = clear_text_list([text])
    vector = vectorized.transform(cleaned_text)
    predicted_class = model.predict(vector)
    return predicted_class[0]


# df = pd.read_csv('dataset/test.csv')
# for index, row in df.iterrows():
#     new_text = row['text']
#     predicted_class = predict_class(new_text)
#     if predicted_class != row['class']:
#         print(predicted_class, row['class'], index)
