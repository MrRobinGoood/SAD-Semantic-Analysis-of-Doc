from striprtf.striprtf import rtf_to_text
import os
import pandas as pd

folders = [
    './dataset/application',
    './dataset/arrangement',
    './dataset/contract',
    './dataset/order',
    './dataset/statute'
]

classes_name = []
texts_of_doc = []
a =0
for folder in folders:
    files = os.listdir(folder)
    for file in files:
        a +=1
        file_path = os.path.join(folder, file)

        # Чтение RTF файла
        with open(file_path, 'r') as file:
            rtf_content = file.read()

        # Преобразование RTF содержимого в обычный текст
        text = rtf_to_text(rtf_content)
        classes_name.append(os.path.split(folder)[-1])
        texts_of_doc.append(text)

dict = {'class': classes_name, 'text': texts_of_doc}
print(len(dict))
df = pd.DataFrame(dict)
df.to_csv('dataset/test.csv', index=False)
