import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Названия кампусов
campuses = ["Москва", "Нижний Новгород", "Санкт-Петербург", "Пермь"]

# Уровни образования
education_levels = ["бакалавриат", "магистратура", "специалитет", "аспирантура"]

# Категории вопросов
question_categories = [
    "Деньги",
    "Учебный процесс",
    "Практическая подготовка",
    "ГИА",
    "Траектории обучения",
    "Английский язык",
    "Цифровые компетенции",
    "Перемещения студентов / Изменения статусов студентов",
    "Онлайн-обучение",
    "Цифровые системы",
    "Обратная связь",
    "Дополнительное образование",
    "Безопасность",
    "Наука",
    "Социальные вопросы",
    "ВУЦ",
    "Общежития",
    "ОВЗ",
    "Внеучебка",
    "Выпускникам",
    "Другое"
]

# Генерация случайных значений для метрик
metric_values = {
    "context_precision": {education: {campus: np.random.rand(6) for campus in campuses} for education in education_levels},
    "answer_correctness_neural": {education: {campus: np.random.rand(6) for campus in campuses} for education in education_levels},
    "answer_correctness_literal": {education: {campus: np.random.rand(6) for campus in campuses} for education in education_levels},
    "context_recall": {education: {campus: np.random.rand(6) for campus in campuses} for education in education_levels}
}

# Расчет общих показателей для каждой метрики
overall_metric_values = {
    metric: {education: np.mean(list(values.values()), axis=0) for education, values in metric_values[metric].items()} for metric in metric_values.keys()
}

# Заголовок приложения
st.title("Визуализация значений метрик")

# Выбор метрики
selected_metric = st.selectbox("Выберите метрику", list(metric_values.keys()))

# Выбор уровня образования
selected_education_level = st.selectbox("Выберите уровень образования", education_levels)

# Выбор кампуса
selected_campus = st.selectbox("Выберите кампус", campuses)

# Отображение графика для выбранной метрики, уровня образования и кампуса
fig, ax = plt.subplots()

# График для выбранной метрики
ax.bar(range(len(metric_values[selected_metric][selected_education_level][selected_campus])),
       metric_values[selected_metric][selected_education_level][selected_campus], color='blue')
ax.set_ylim(0, 1.1)
ax.set_ylabel("Значение")
ax.set_title(f"{selected_metric} для {selected_education_level} в {selected_campus}")
ax.set_xticks(range(len(metric_values[selected_metric][selected_education_level][selected_campus])))
ax.set_xticklabels([f"Значение {i + 1}" for i in range(len(metric_values[selected_metric][selected_education_level][selected_campus]))])

# Отображение графика
st.pyplot(fig)

# Отображение значений для всех кампусов по каждой метрике для выбранного уровня образования
st.write(f"Значения для метрики {selected_metric} для каждого кампуса на уровне {selected_education_level}:")
for campus, values in metric_values[selected_metric][selected_education_level].items():
    st.write(f"{campus}: {values.tolist()}")

# Отображение общих показателей по каждой метрике для выбранного уровня образования
st.write(f"Общие показатели по метрике {selected_metric} для {selected_education_level}:")
st.write(f"Средние значения: {overall_metric_values[selected_metric][selected_education_level].tolist()}")

# Дополнительное использование выбранной категории вопросов
selected_question_category = st.selectbox("Выберите категорию вопросов", question_categories)
st.write(f"Выбранная категория вопросов: {selected_question_category}")
