import streamlit as st
import json
import time
import pandas as pd

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


def update_counts(campus, education, category, response_time, chat_history):
    global empty_chat_history_count, non_empty_chat_history_count

    campus_counts[campus] = campus_counts.get(campus, 0) + 1
    education_counts[education] = education_counts.get(education, 0) + 1
    category_counts[category] = category_counts.get(category, 0) + 1
    response_times.append(response_time)

    if chat_history:
        non_empty_chat_history_count += 1
    else:
        empty_chat_history_count += 1


# Контейнеры для обновления графиков и метрик

st.header("Графики")

campus_chart = st.empty()
edu_chart = st.empty()
category_chart = st.empty()

st.header("Производительность")

response_time_text = st.empty()
chat_history_empty = st.empty()
chat_history_not_empty = st.empty()


# Читаем JSON построчно
with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

    for entry in data:
        campus = entry.get("Кампус", "Неизвестно")
        education = entry.get("Уровень образования", "Неизвестно")
        category = entry.get("Категория вопроса", "Неизвестно")
        response_time = entry.get("Время ответа модели (сек)", 0)
        chat_history = entry.get("Уточненный вопрос пользователя", "")

        update_counts(campus, education, category, response_time, chat_history)

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

        # Пауза в 1 секунду
        time.sleep(1)
