import torch
from transformers import BertForSequenceClassification, Trainer, TrainingArguments, BertTokenizerFast
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd

# Загрузка датасета
df = pd.read_csv('../dataset/sample.csv')

# Разделение данных на обучающую и тестовую выборки
train_texts, test_texts, train_labels, test_labels = train_test_split(df['text'], df['class'], test_size=0.01, random_state=42)

from sklearn.preprocessing import LabelEncoder

# Инициализация токенизатора
tokenizer = BertTokenizerFast.from_pretrained('rubert-base-cased')

# Токенизация текстов с указанием максимальной длины
train_encodings = tokenizer(train_texts.tolist(), truncation=True, padding=True, max_length=512)
test_encodings = tokenizer(test_texts.tolist(), truncation=True, padding=True, max_length=512)

# Создание даталоадеров
class DocumentDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        # Преобразование строки в числовое значение и изменение типа данных на Long
        item['labels'] = torch.tensor(self.labels[idx]).long()
        return item

    def __len__(self):
        return len(self.labels)

# Преобразование строк в числовые значения
label_encoder = LabelEncoder()
train_labels_encoded = label_encoder.fit_transform(train_labels)
test_labels_encoded = label_encoder.transform(test_labels) # Используйте transform, а не fit_transform, чтобы избежать повторного обучения

train_dataset = DocumentDataset(train_encodings, train_labels_encoded)
test_dataset = DocumentDataset(test_encodings, test_labels_encoded)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
print('1')
# Инициализация модели
model = BertForSequenceClassification.from_pretrained('rubert-base-cased', num_labels=11)
print('2')
# Определение аргументов для обучения
training_args = TrainingArguments(
    output_dir='../results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='../logs',
    logging_steps=10, # Логирование каждые 10 шагов
    report_to=["tensorboard"], # Использование TensorBoard для логирования
)

# Создание тренера
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# Обучение модели с логированием прогресса
trainer.train()

# Сохранение модели после обучения
trainer.save_model("./my_model")
print("Модель сохранена.")

print('3')
# Оценка модели на тестовом наборе данных
trainer.evaluate()

# Пример предсказания
text = "Пример текста для классификации"
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)
predictions = torch.argmax(outputs.logits, dim=1)

# Вывод предсказания
print(f"Предсказанный класс: {predictions.item()}")

# # Получение списка всех уникальных классов и их соответствующих номеров
# class_names = label_encoder.classes_
# class_numbers = list(range(len(class_names)))
#
# # Вывод соответствия классов и номеров
# for class_name, class_number in zip(class_names, class_numbers):
#     print(f"Класс: {class_name}, Номер: {class_number}")