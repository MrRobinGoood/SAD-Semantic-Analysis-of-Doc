import os
import pandas as pd
from striprtf.striprtf import rtf_to_text
from PyPDF2 import PdfReader
from docx import Document
#import textract

folders = [
    './new_dataset/act',
    './new_dataset/application',
    './new_dataset/arrangement',
    './new_dataset/contract',
    './new_dataset/contract offer',
    './new_dataset/determination',
    './new_dataset/invoice',
    './new_dataset/order',
    './new_dataset/proxy',
    './new_dataset/statute'
]

classes_name = []
texts_of_doc = []

for folder in folders:
    files = os.listdir(folder)
    for file in files:
        file_path = os.path.join(folder, file)
        file_extension = file.split('.')[-1]

        try:
            if file_extension == "rtf":
                # Чтение RTF файла
                with open(file_path, 'r') as file:
                    rtf_content = file.read()
                # Преобразование RTF содержимого в обычный текст
                text = rtf_to_text(rtf_content)
            elif file_extension == "pdf":
                # Чтение PDF файла
                pdf_reader = PdfReader(file_path)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()
            elif file_extension == "docx":
                # Чтение DOCX файла
                document = Document(file_path)
                text = " ".join([paragraph.text for paragraph in document.paragraphs])
            elif file_extension == "doc":
                pass
                # Чтение DOC файла
                #text = textract.process(file_path).decode('utf-8')
            else:
                continue # Пропускаем файлы других форматов

            classes_name.append(os.path.split(folder)[-1])
            texts_of_doc.append(text)
        except Exception as e:
            print(f"Ошибка при чтении файла {file_path}: {e}")

dict = {'class': classes_name, 'text': texts_of_doc}
print(len(dict))
df = pd.DataFrame(dict)
df.to_csv('dataset/test.csv', index=False)
