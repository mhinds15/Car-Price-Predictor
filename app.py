import pandas as pd
import streamlit as st
import joblib
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error, r2_score




pipline = joblib.load('best_pipeline_model.pkl')

df = pd.read_csv('data/merc.csv')
X=df.drop(columns='price')
y = df['price']

def main():
    st.title("Car Price Prediction")
    X=df.drop(columns='price')
    y = df['price']

    
    model = st.selectbox('Model', ['A Class', 'B Class', 'C Class','G Class', 'GLE Class', 'GLA Class',
                                   'CLA Class', 'S Class', 'SL Class', 'GLS Class'])
    transmission = st.selectbox('Transmission', ['Manual', 'Automatic', 'Semi-Auto'])
    fuelType = st.selectbox('Fuel Type', ['Petrol', 'Diesel', 'Hybrid'])
    year = st.number_input('Year', min_value=1990, max_value=2025, value=2022)
    mileage = st.number_input('Mileage', min_value=0, max_value=1000000, value=0)
    tax = st.number_input('Road Tax', min_value=0, value=150)
    mpg = st.number_input('MPG', min_value=0.0, value=60.0)
    engineSize = st.number_input('Engine Size', min_value=0.0, value=1.5)

    new_data = pd.DataFrame({
        'model': [model],
        'year': [year],
        'transmission': [transmission],
        'mileage': [mileage],
        'fuelType': [fuelType],
        'tax': [tax],
        'mpg': [mpg],
        'engineSize': [engineSize],
    })

    if st.button('Predict'):
        prediction = pipline.predict(new_data)[0]
        st.success(f'Estimated Price: £{prediction:,.0f}')


    fig = px.scatter(df, x='mileage', y='price', color='fuelType', title='Mileage vs Price')
    st.plotly_chart(fig)

    y_pred = pipline.predict(X)
    results_df = pd.DataFrame({
        'Actual Price': y,
        'Predicted Price': y_pred
    })

    fig2= px.scatter(
        results_df,
        x='Actual Price',
        y='Predicted Price',
        title='Predicted vs Actual Car Prices',
        labels={'Actual Price': 'Actual Price (£)', 'Predicted Price': 'Predicted Price (£)'}, 
        template='plotly_white'
    )

    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)

    st.markdown('Model Performance Metrics')
    col1, col2, col3 = st.columns(3)

    col1.metric('MAE', f'£{mae:,.0f}')
    col2.metric('RMSE', f'£{rmse:,.0f}')
    col3.metric('R2 Score', f'{r2:,.2f}')

    st.plotly_chart(fig2, use_container_width=True)

    fig3 = px.box(df, x='transmission', y='price', title='Price by Transmission Type')
    st.plotly_chart(fig3)


if __name__ == '__main__':
    main()