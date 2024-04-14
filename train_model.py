import pandas as pd
from joblib import dump
from prepare_text import clear_text_list
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC

df = pd.read_csv("dataset/sample.csv")

vectorized = CountVectorizer()

X = vectorized.fit_transform(clear_text_list(list(df['text'])))
y = df['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = SVC(kernel='linear')

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

f1 = f1_score(y_test, y_pred, average='macro')

dump(model, 'model/temp1.joblib')

dump(vectorized, 'model/temp2.joblib')

print(f"F1-score: {f1}")
