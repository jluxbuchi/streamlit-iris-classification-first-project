
import streamlit as st

import numpy as np

import pickle

 

model =  pickle.load(open('model.pkl'  , "rb"))

scaler = pickle.load(open('scaler.pkl' , "rb"))

st.title(" ~ Iris Flower Classification  ~ ")

 

sepal_len = st.number_input(" Sepal Length ")

sepal_wd  = st.number_input(" Sepal Width ")

petal_len = st.number_input(" Petal Length ")

petal_wd  = st.number_input(" Petal Width ")

 

if st.button("Predict") :

    data = np.array([[sepal_len , sepal_wd, petal_len, petal_wd]])

    data = scaler.transform(data)

    prediction = model.predict(data)

 

    species = ['Setosa', 'Versicolor' , 'Virginica']

    st.success(species[prediction[0]])