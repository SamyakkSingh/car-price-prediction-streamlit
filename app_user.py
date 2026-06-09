import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title='Car Price Prediction', layout='centered')

st.title('Car Price Prediction')

@st.cache_resource
def load_model():
    return joblib.load('car_price_model_user.joblib')

model = None
try:
    model = load_model()
except Exception as e:
    st.warning('Model not found or failed to load. Run train_model.py or the notebook to create car_price_model_user.joblib')

# If dataset exists, try to populate select boxes
try:
    df = pd.read_csv('car_data.csv')
    car_names = sorted(df['Car_Name'].dropna().unique().tolist())
    fuel_types = sorted(df['Fuel_Type'].dropna().unique().tolist())
    transmissions = sorted(df['Transmission'].dropna().unique().tolist())
    owners = sorted(df['Owner'].dropna().unique().tolist())
except Exception:
    car_names = []
    fuel_types = []
    transmissions = []
    owners = []

present_price = st.number_input('Ex-showroom price when the owner bought it (lakhs)', min_value=0.0, value=5.0, step=0.1)
driven_kms = st.number_input('Driven Kms', min_value=0, max_value=1000000, value=50000, step=500)
age = st.slider('Age (years)', 0, 30, 5)

if car_names:
    car_name = st.selectbox('Car Name', car_names)
else:
    car_name = st.text_input('Car Name')

if fuel_types:
    fuel = st.selectbox('Fuel Type', fuel_types)
else:
    fuel = st.selectbox('Fuel Type', ['Petrol', 'Diesel', 'CNG', 'Hybrid', 'Electric'])

if transmissions:
    trans = st.selectbox('Transmission', transmissions)
else:
    trans = st.selectbox('Transmission', ['Manual', 'Automatic'])

if owners:
    owner = st.selectbox('Owner', owners)
else:
    owner = st.selectbox('Owner', [0, 1, 2, 3])

if st.button('Predict'):
    if model is None:
        st.error('Model not loaded. Run the notebook or train_model.py to create car_price_model_user.joblib in this folder.')
    else:
        X = pd.DataFrame([{
            'Present_Price': present_price,
            'Driven_kms': driven_kms,
            'Age': age,
            'Car_Name': car_name,
            'Fuel_Type': fuel,
            'Transmission': trans,
            'Owner': owner
        }])
        # Ensure column names ordering doesn't matter: model pipeline handles columns by name via OneHotEncoder transform
        pred = model.predict(X)[0]
        st.success(f'Estimated Selling Price: {pred:.2f}')
