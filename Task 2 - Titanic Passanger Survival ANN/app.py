import streamlit as st
import numpy as np
import os
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import tensorflow as tf


# Reading the Dataset for Productivity 
df = pd.read_csv("Data/titanic.csv")
df = df.dropna()
# st.dataframe(df)
# st.write(df.columns)
model = tf.keras.models.load_model('Models/titan_titanic_train_500_32_8_relu_relu_1692464561')




# Building the App


st.title("Titanic Passanger Survival Prediction :ship:")
id = st.number_input("Enter Passanger ID")
Pclass = st.selectbox("Enter Passanger Class", options=list(sorted(df["Pclass"].unique())))
Name = st.text_input("Enter Name")
Gender = st.selectbox("Enter Gender", options=list(df["Sex"].unique()))
Age = st.number_input("Enter Age")
SipSp = st.selectbox("Enter Sibsip", options=list(df["SibSp"].unique()))
Porch = st.selectbox("Select Porch", options=list(df["Parch"].unique()))
Ticket = st.selectbox("Select the Ticket", options=list(df["Ticket"].unique()))
Fare = st.number_input("Enter the Fare")
Cabin = st.selectbox("Select the Cabin", options=list(df["Cabin"].unique()))
Embarked = st.selectbox("Select Embarked", options=list(df["Embarked"].unique()))

columns = ['PassengerId',  'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
processor = joblib.load("objects/processor.joblib")


row = np.array([id, Pclass, Gender, Age, SipSp, Porch, Ticket, Fare, Cabin, Embarked])
# st.write(row)
X = pd.DataFrame([row], columns=columns)
# st.write(X)
t = processor.transform(X)
st.write(t)



def predict():
    row = np.array([id, Pclass, Gender, Age, SipSp, Porch, Ticket, Fare, Cabin, Embarked])
    X = pd.DataFrame([row], columns=columns)
    trains = processor.transform(X)
    prediction = model.predict(trains)
    if prediction > 0.5:     
        st.success("The Person Will Survive")
    else:
        st.success("The Person Will Npt Survive")
    return prediction

if st.button('Price Prediction', on_click=predict):
        predict()