import pandas as pd
import streamlit as st

import altair as alt
import plotly.express as px


if __name__ == '__main__':

    st.set_page_config(
        page_title="test_dataset",
        layout="wide")

    alt.themes.enable("dark")
    df = pd.read_json("/home/guest/Documents/hakaton/hackathon_hse25/prepocess_calculate/datasets/train_set.json")
    print(df.head())

    hm_data=df.groupby(['Кампус', 'Категория вопроса']).size().reset_index(name='Count')
    '''
    heatmap = alt.Chart(hm_data).mark_rect().encode(
            y=alt.Y(f'Кампус:O', axis=alt.Axis(title="Кампус", titleFontSize=18, titlePadding=15, titleFontWeight=900, labelAngle=0)),
            x=alt.X(f'Категория вопроса:O', axis=alt.Axis(title="Категория вопроса", titleFontSize=18, titlePadding=15, titleFontWeight=900)),
            color=alt.Color(f'max(magenta):Q',
                             legend=None,
                             scale=alt.Scale(scheme='blues')),
            tootlip=['Кампус','Категория вопроса','Count']).properties(width=900
        ).configure_axis(
        labelFontSize=12,
        titleFontSize=12
        )
        '''
    heatmap = alt.Chart(hm_data).mark_rect().encode(
        y=alt.Y('Кампус:O', axis=alt.Axis(title="Кампус", titleFontSize=18, titlePadding=15, titleFontWeight=900)),
        x=alt.X('Категория вопроса:O',
                axis=alt.Axis(title="Категория вопроса", titleFontSize=18, titlePadding=15, titleFontWeight=900)),
        color=alt.Color('Count:Q', scale=alt.Scale(scheme='blues'), title='Count'),
        tooltip=['Кампус', 'Категория вопроса', 'Count']
    ).properties(
        width=800,
        height=400
    ).configure_axis(
        labelFontSize=12,
        titleFontSize=14
    )
    st.altair_chart(heatmap, use_container_width=True)
