import pandas as pd

# # Пути к вашим CSV-файлам
# csv_file_1 = 'dataset/sample.csv'
# csv_file_2 = 'dataset/test.csv'
#
# # Считывание CSV-файлов в датафреймы
# df1 = pd.read_csv(csv_file_1)
# df2 = pd.read_csv(csv_file_2)
#
# # Объединение датафреймов
# # Используйте concat для объединения по столбцам
# # или append для объединения по строкам
# combined_df = pd.concat([df1, df2], ignore_index=True)
#
# # Сохранение объединенного датафрейма в новый CSV-файл
# combined_df.to_csv('all_dataset.csv', index=False)

df1 = pd.read_csv('dataset/all_dataset.csv')

print(df1.head(-1))