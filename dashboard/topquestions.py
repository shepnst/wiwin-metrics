



import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


top_quest={"как добраться до корпуса":60, "где есть столовая":40, "как получить справку":35, "что такое СОП":28, "кто директор?":10}


data = pd.DataFrame(list(top_quest.items()), columns=['Вопрос', 'Количество'])
data.set_index('Вопрос', inplace=True)

plt.figure(figsize=(8, 4))
sns.heatmap(data.T, annot=True, cmap='YlGnBu', cbar=True, fmt='g')

plt.title("Самые часто задаваемые вопросы")

st.pyplot(plt)
