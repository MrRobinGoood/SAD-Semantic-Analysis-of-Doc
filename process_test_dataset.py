import pandas as pd
from model import predict_class
df = pd.read_csv('test_dataset/dataset.csv', delimiter="|")

dd = {"document_id":[],"class_id":[]}
for index, row in df.iterrows():
    new_text = row['document_text']
    predicted_class = predict_class(new_text)
    dd["document_id"].append(row['document_id'])
    dd["class_id"].append(predicted_class)

df2 = pd.DataFrame(dd)


df2['class_id'] = df2['class_id'].map({'proxy':1,'contract':2,'act':3,'application':4,'order':5,'invoice':6,'bill':7,'arrangement':8,'contract offer':9,'statute':10,'determination':11})
df2.to_csv('result/submission2.csv', index=False, sep=';')

print(df2.head(-1))