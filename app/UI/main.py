'''UI Module'''
import json
import requests
import streamlit as st

st.write("""
# Application to predict the time for the NYC taxi trips
""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    with st.sidebar:
        pu = st.text_input('PU Location ID')
        do = st.text_input('DO Location ID')
        td = st.number_input('Trip Distance', value=10.0, min_value=1.0, max_value=100.0)

    data = {'PULocationID': pu,
            'DOLocationID': do,
            'trip_distance': td}
    return data

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

if st.button('Predict'):
    response = requests.post(url = 'http://nyc-taxi-model-container:8000/predict',
                             data = json.dumps(df),
                             timeout=100)
    x = type(response.text)
    st.write(f'El tiempo estimado de viaje es de {response.json()["prediction"]} minutos')
