import streamlit as st
import matplotlib.pyplot as plt
import json

# Загрузка метрик из файла
try:
    with open('metrics.json', 'r', encoding='utf-8') as f:
        metrics = json.load(f)
        general_score = metrics.get('general_score', 0.68)
        recall = metrics.get('recall', 0.96)  # Используем context_recall как recall
        precision = metrics.get('precision', 0.32)  # Используем context_precision как precision
        answer_correctness_literal = metrics.get('answer_correctness_literal', 0.56)
        answer_correctness_neural = metrics.get('answer_correctness_neural', 0.74)
except FileNotFoundError:
    # Если файл не найден, используем значения по умолчанию
    general_score = 0.68
    recall = 0.96
    precision = 0.32
    answer_correctness_literal = 0.56
    answer_correctness_neural = 0.74

def plot_donut(value, title, ax):
    sizes = [value, 1 - value]
    colors = ['#00008B', '#f0f0f0']
    labels = [f'{value * 100:.0f}%', '']

    ax.pie(sizes, colors=colors, labels=labels, startangle=90, wedgeprops={'width': 0.4})

    centre_circle = plt.Circle((0, 0), 0.2, color='white', linewidth=0)
    ax.add_artist(centre_circle)

    ax.set_title(title, fontsize=14, pad=10)

st.title("Метрики")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

axes = axes.flatten()

plot_donut(general_score, "General Score", axes[0])
plot_donut(recall, "Recall", axes[1])
plot_donut(precision, "Precision", axes[2])
plot_donut(answer_correctness_literal, "Answer Correctness (Literal)", axes[3])
plot_donut(answer_correctness_neural, "Answer Correctness (Neural)", axes[4])

axes[5].axis('off')

plt.tight_layout()

st.pyplot(fig)