import streamlit as st
import json
import time
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Заголовок Streamlit
st.title("Аналитика запросов студентов ВШЭ")

# Загружаем файл
file_path = "datasets/val_set.json"

# Структуры данных для анализа
campus_counts = {}
education_counts = {}
category_counts = {}
response_times = []
empty_chat_history_count = 0
non_empty_chat_history_count = 0
questions = []


def update_counts(campus, education, category, response_time, chat_history, question):
    global empty_chat_history_count, non_empty_chat_history_count

    campus_counts[campus] = campus_counts.get(campus, 0) + 1
    education_counts[education] = education_counts.get(education, 0) + 1
    category_counts[category] = category_counts.get(category, 0) + 1
    response_times.append(response_time)
    questions.append(question)

    if chat_history:
        non_empty_chat_history_count += 1
    else:
        empty_chat_history_count += 1


def find_duplicate_questions():
    if len(questions) < 2:
        return 0

    vectorizer = TfidfVectorizer().fit_transform(questions)
    similarity_matrix = cosine_similarity(vectorizer)

    duplicate_count = 0
    threshold = 0.8  # Порог для определения схожести вопросов
    for i in range(len(questions)):
        for j in range(i + 1, len(questions)):
            if similarity_matrix[i, j] > threshold:
                duplicate_count += 1

    return duplicate_count


# Контейнеры для обновления графиков и метрик

st.header("Графики")

campus_chart = st.empty()
edu_chart = st.empty()
category_chart = st.empty()

st.header("Производительность")

response_time_text = st.empty()
chat_history_empty = st.empty()
chat_history_not_empty = st.empty()

st.header("История чатов")
duplicates_text = st.empty()


# Читаем JSON построчно
with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

    for entry in data:
        campus = entry.get("Кампус", "Неизвестно")
        education = entry.get("Уровень образования", "Неизвестно")
        category = entry.get("Категория вопроса", "Неизвестно")
        response_time = entry.get("Время ответа модели (сек)", 0)
        chat_history = entry.get("Уточненный вопрос пользователя", "")
        question = entry.get("Вопрос пользователя", "")

        update_counts(campus, education, category, response_time, chat_history,question)

        # Обновляем графики
        campus_df = pd.DataFrame(list(campus_counts.items()), columns=["Кампус", "Количество"])
        edu_df = pd.DataFrame(list(education_counts.items()), columns=["Уровень образования", "Количество"])
        category_df = pd.DataFrame(list(category_counts.items()), columns=["Категория", "Количество"])

        campus_chart.bar_chart(campus_df.set_index("Кампус"))
        edu_chart.bar_chart(edu_df.set_index("Уровень образования"))
        category_chart.bar_chart(category_df.set_index("Категория"))

        # Вычисляем среднее время ответа
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        response_time_text.subheader(f"Среднее время обработки вопросов: {avg_response_time:.2f} сек")

        # Отображаем статистику по chat_history
        chat_history_empty.subheader(f"✅Пользователям понравился ответ: {empty_chat_history_count}")
        chat_history_not_empty.subheader(f"❌Пользователям не понравился ответ: {non_empty_chat_history_count}")

        # Анализ повторяющихся вопросов
        duplicate_count = find_duplicate_questions()
        duplicates_text.subheader(f"Частота повторяющихся вопросов: {duplicate_count}")

        # Пауза в 1 секунду
        time.sleep(1)
