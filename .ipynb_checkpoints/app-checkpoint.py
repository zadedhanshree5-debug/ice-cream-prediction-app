import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pickle

with open("model.pkl","rb") as f:
    model = pickle.load(f)

# title
st.title("Ice Cream Sales Prediction App")
st.write("Predict ice cream sales based om temprature")


# input from user
temp = st.number_input("Enter Temperature (°C)", value =25.0)


 # predict button
if st.button("Predict Sales"):
   input_data = np.array([[temp, temp**2]])
   prediction = model.predict(input_data)
   prediction_value = prediction[0] # extract actual number
   st.success(f"Predicted Ice Cream Sales: {prediction_value:2f} units")

#-----------------------------------------
# Show Dataset + Graph
#--------------------------------------
st.subheader("Dataset Visualization")

df = pd.read_csv("IceCreamSalesData.csv")

fig, ax = plt.subplots()
ax.scatter(df['Temperature (°C)'],df['Ice Cream Sales (units)'])
ax.set_xlabel("Temperature (°C)")
ax.set_ylabel("Sales")

st.pyplot(fig)
