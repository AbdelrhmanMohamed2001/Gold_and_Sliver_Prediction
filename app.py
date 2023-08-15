import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import streamlit as st
from PIL import Image


dataset= pd.read_csv('gld_price_data.csv')
image= Image.open('gold_silver.jpg')
st.title("PREDICTION OF GOLD AND SILVER PRICES")
st.image(image, use_column_width=True)
Prediction= st.checkbox("Prediction")
Data_Analysis= st.checkbox("SHOW DATA ANALYSIS")

if Data_Analysis:

  st.write("**Head of Dataset:** ", dataset.head())
  st.write("**DESCRIPTION of Dataset:** ", dataset.describe())
  st.write("**CHECH NULL VALUES:** ", dataset.isnull().sum())
  st.write("**CHECH DUPLICATED VALUES:** ", dataset.duplicated().sum())
  for column in dataset.columns:
    unique_values = dataset[column].unique()
    if unique_values.size > 0:
        st.write(column, "**has**", unique_values.size, "**unique values**")
    else:
        st.write(column, "**doesn't have any unique values**")
  for column in dataset.columns:
    st.write(column)
    st.write(dataset[column].value_counts())
    st.write(dataset[column].value_counts().sum())


  st.write("**CORRELATION OF THIS DATASET:** ")
  corr_matrix = dataset.corr()
  fig, ax = plt.subplots()
  sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
  st.pyplot(fig)

  correlation = dataset.corr()
  for column in dataset.columns:
    if column == "Date":
      st.write("**Date haven't correlation**")
    else:
      st.write("correlation of ",column, ":")
      st.write(correlation[column])


  for column in dataset.columns:
    if column=="Date":
       print("**NO DISTRIBUTION FOR DATE **")
    else:

       fig, ax = plt.subplots()
       ax.hist(dataset[column], bins=30)
       ax.set_title(f"**Distribution of {column}**")
       ax.set_xlabel("Value")
       ax.set_ylabel("Frequency")
       st.pyplot(fig)


else:

    if Prediction :
       Gold_Prediction= st.checkbox("Gold Prediction")
       Silver_Prediction= st.checkbox("Silver Prediction")

       if Gold_Prediction:

         XG=dataset[['SPX', 'USO', 'SLV', 'EUR/USD']]
         YG=dataset['GLD']
         SPX=st.number_input("Enter SPX")
         USO=st.number_input("Enter USO")
         SLV=st.number_input("Enter SLV")
         EUR_USD=st.number_input("Enter EUR_USD")
         model = LinearRegression()
         model.fit(XG, YG)
         prediction_G=model.predict([[SPX, USO, SLV, EUR_USD]])[0]
         if st.button("PREDICT GOLD PRICE"):
             st.header("The Gold Price is {}".format(int(prediction_G)))
         gold_accuracy = model.score(XG, YG)
         st.write("Accuracy is: ", gold_accuracy, "With Linear Regression Model ")



       else:
         XS=dataset[['SPX', 'USO', 'GLD', 'EUR/USD']]
         YS=dataset['SLV']
         SPX=st.number_input("Enter SPX")
         USO=st.number_input("Enter USO")
         GLD=st.number_input("Enter GLD")
         EUR_USD=st.number_input("Enter EUR_USD")
         model = LinearRegression()
         model.fit(XS, YS)
         prediction_S=model.predict([[SPX, USO, GLD, EUR_USD]])[0]
         if st.button("PREDICT SILVER PRICE"):
             st.header("The Silver Price is {}".format(int(prediction_S)))

         silver_accuracy = model.score(XS, YS)
         st.write("Accuracy is: ", silver_accuracy, "With Linear Regression Model")



