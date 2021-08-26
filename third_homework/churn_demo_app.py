import streamlit as st
import pandas as pd
import numpy as np
import joblib
from churn_classifier import ChurnClassifier
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
# import re
# from PIL import Image


df = joblib.load('dataset.pkl')

st.title('Предсказание оттока клиентов')

st.markdown('Это **демонстрационный стенд** нашей модели:')
col1, col2, col3 = st.columns(3)
col1.metric("Roc auc", "0.93")
col2.metric("Precision", "0.89")
col3.metric("Recall", "0.67")
st.markdown('\nВводимый датасет должен состоять из следующих полей:')

np.random.seed(42)
n_churn = len(df[df['Churn']==1])
noise = df[df['Churn']==0]['Account length'].max() // 3
df['Lifetime'] = df['Account length']
df.loc[df['Churn']==1, ['Lifetime']] = df[df['Churn']==1]['Lifetime'] + pd.Series(np.round(np.random.uniform(0, noise, n_churn*8)))
df['Lifetime'] = df['Lifetime'].astype("int64")
df[df['Churn']==1].head()

df = df.drop(columns={'Churn', 'Total charge'})

with st.form('\nform'):
    link_input = st.text_area('Вставьте ссылку в окно ниже: ', 'вставьте ссылку сюда...')
    option = st.selectbox('Выберите метрику',
    ('Отток(в %)', 'Отток(количество)', 'Не отток(в %)', 'Не отток(количество)'))
    selected_states = st.multiselect("Выберите штаты", df['State'].unique())
    submit_button = st.form_submit_button('Сделать предсказание')


if submit_button:
    df_to_predict = pd.read_csv(link_input)
    model = ChurnClassifier()
    predict = model.predict_customer_churn(df_to_predict)
    st.markdown(f"<h3 style='text-align: center; color: black;'>Статистика по всему набору данных </h3>"
                f"<h5><br> Отток: {round(len(predict[predict.Churn==True]) / len(predict)*100)} %,"
                f"{predict[predict.Churn==True]['Churn'].sum()} человек"
                f"<br> Не отток: {round((1 - predict[predict.Churn==True]['Churn'].sum() / len(predict)) * 100)} %,"
                f" {len(predict) - predict[predict.Churn==True]['Churn'].sum()} человек"
                f"</h5>",
                unsafe_allow_html=True)
    encoder = joblib.load('enc.pkl')
    data = []
    if option == 'Отток(в %)':
        for i in selected_states:
            data.append(100 * predict[predict['State'] == encoder.transform([i])[0]]['Churn'].sum() / len(predict[predict['State'] == encoder.transform([i])[0]]))
        fig, ax = plt.subplots()
        ax.bar(selected_states, data, width=0.5)
        st.pyplot(fig)

    elif option == 'Отток(количество)':
        for i in selected_states:
            data.append(predict[predict['State'] == encoder.transform([i])[0]]['Churn'].sum())
        fig, ax = plt.subplots()
        ax.bar(selected_states, data, width=0.5)
        st.pyplot(fig)

    elif option == 'Не отток(в %)':
        for i in selected_states:
            data.append(1 - predict[predict['State'] == encoder.transform([i])[0]]['Churn'].sum() / len(predict[predict['State'] == encoder.transform([i])[0]]))
        fig, ax = plt.subplots()
        ax.bar(selected_states, data, width=0.5)
        st.pyplot(fig)

    elif option == 'Не отток(количество)':
        for i in selected_states:
            data.append(len(predict[predict['State'] == encoder.transform([i])[0]]) - predict[predict['State'] == encoder.transform([i])[0]]['Churn'].sum())
        fig, ax = plt.subplots()
        ax.bar(selected_states, data, width=0.5)
        st.pyplot(fig)







with st.expander("Посмотреть объяснение"):
    st.write("""
         The chart above shows some numbers I picked for you.
         I rolled actual dice for these, so they're *guaranteed* to
         be random.
                """)
    st.image("https://static.streamlit.io/examples/dice.jpg")
# st.write("Выбранные штаты:", selected_states)
