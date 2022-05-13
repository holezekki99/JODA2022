import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
import pickle

st.markdown('## Salary Predictor For NHL Goalies')
model = pickle.load(open('../model.pkl', 'rb')) # load the model

# Take the users input
goals_pg = st.number_input('Goals per game')
rebounds_pg = st.number_input('Rebounds per game')
lowDangerxGoals_pg = st.number_input('Goals allowed for low danger shots per game')
mediumDangerxGoals_pg = st.number_input('Goals allowed for medium danger shots per game')
highDangerxGoals_pg = st.number_input('Goals allowed for high danger shots per game')

# store the inputs
features = [goals_pg, rebounds_pg, lowDangerxGoals_pg, mediumDangerxGoals_pg, highDangerxGoals_pg]
# convert user inputs into an array fr the model
#int_features = [int(x) for x in features]
#final_features = [np.array(int_features)]

# convert user inputs into an array fr the model
#int_features = [int(x) for x in features]
final_features = [np.array(features)]

if st.button('Predict'): # when the submit button is pressed
    prediction = model.predict(final_features)
    st.balloons()
    st.success(f'Goalie Salary per season is:  {round(prediction[0])} $')