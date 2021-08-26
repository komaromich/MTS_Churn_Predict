import streamlit as st
import pandas as pd
import joblib
from churn_classifier import ChurnClassifier
from PIL import Image

# df = pd.DataFrame({
#   'first column': [1, 2, 3, 4],
#   'second column': [10, 20, 30, 40]
# })
#
# df


df = joblib.load('dataset.pkl')

list_of_fields = ['Account length', 'Area code', 'International plan', 'Voice mail plan',
       'Number vmail messages', 'Total day calls', 'Total day charge',
       'Total eve calls', 'Total eve charge', 'Total night calls',
       'Total night charge', 'Total intl calls', 'Total intl charge',
       'Customer service calls']

st.title('Предсказание оттока клиентов')

st.markdown('Это **демонстрационный стенд**')
st.markdown('Вводимый датасет должен состоять из следующих полей:')
# st.markdown(list_of_fields)
df[df.index < 10]
image = Image.open('df_image.png')
st.image(image, caption="Пример dataframe'a")

with st.form('form'):
	link_input = st.text_area('Вставьте ссылку в окно ниже: ', 'встаьте ссылку сюда...')
	submit_button = st.form_submit_button('Сделать предсказание')


if submit_button:
    df_to_predict = pd.read_csv(link_input)
    model = ChurnClassifier()
    predict = model.predict_customer_churn(df_to_predict)
    st.write(round(len(df[df.Churn==True]) / len(df[df.Churn==False])*100), 'Процент клиентов,которые покинут вашего телеком оператора')
    st.write(round(df['Churn'].mean()), 'Процент клиентов,которые покинут вашего телеком оператора')

selected_states = st.multiselect("Выберите штаты", df['State'].unique())

# st.write("Выбранные штаты:", selected_states)
