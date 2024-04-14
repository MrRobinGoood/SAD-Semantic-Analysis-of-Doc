# SAD-Semantic-Analysis-of-Documents
### *Веб-приложение для классификации документов по семантическим признакам (backend-part)*
## Содержание

- [Стек технологий](#стек-технологий)
- [Описание проекта](#описание-проекта)
- [Инструкция для запуска](#инструкция-для-запуска)

## Стек технологий
Основной backend-стек:
- Python 3.10
- Scikit-learn
- Pymorphy2
- Nltk
- FastAPI
- Pandas
- SVM
Основной frontend-стек:
- Js
- React
- Redux
## Описание проекта
Данный проект представляет собой веб-приложение обрабатывающее загруженные документы. Наше решение анализирует документ и выдаёт класс, которому соответствует документ.

 - Интерфейс приложения: 

![интерфейс1](https://github.com/MrRobinGoood/SAD-Semantic-Analysis-of-Doc/blob/master/resources/Screenshot_4.png)

![интерфейс2](https://github.com/MrRobinGoood/SAD-Semantic-Analysis-of-Doc/blob/master/resources/Screenshot_5.png)

На данный момент доступны 11 классов:
- proxy - доверенность
- contract - договор
- act - акт
- application - заявление
- order - приказ
- invoice - счет
- bill - приложение
- arrangement - соглашение
- contract offer - договор оферты
- statute - устав
- determination - решение

Наше решение работает с форматами PDF, WORD(.docx) и EXCEL(.xlsx)

Также мы собрали датасет для обучения, в котором более 1100 документов [Датасет](https://disk.yandex.ru/d/0peXuWR-dOxwdg). 

Результат обработки тестового датасета [submission.csv](https://github.com/MrRobinGoood/SAD-Semantic-Analysis-of-Doc/blob/master/result/submission.csv)

## Инструкция для запуска
### Для локального запуска на машине
Пошаговая инструкция:
1. Вам необходимо склонировать репозиторий
2. Для запуска проекта вам потребуется python версии 3.10, а также установленный pip
3. Создайте виртуальное окружение venv и активируйте его
4. Для запуска следующих скриптов перейдите в корневую директорию проекта(по умолчанию Vacancy-Handler-Backend)
5. Запустите командную строку из корневой директории проекта
6. Выполните следующую команду ```pip install -r requirements.txt``` для загрузки всех библиотек-зависимостей
7. Командой  ```uvicorn app:app --reload``` вы можете запустить backend-сервис
8. Вы можете перейти к установке [Frontend](https://github.com/MrRobinGoood/SAD-frontend) части, если еще не сделали этого, либо воспользоваться автоматической документацией Swagger по ссылке ```http://localhost:8000/docs```

Для остановки backend сервиса нажмите комбинацию клавиш ctrl+c или остановите выполняемый процесс
