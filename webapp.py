# Description: This program predicts the annual amount in a Ecommerce website or app.

# Importing the Libraries.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import streamlit as st
from PIL import  Image
import seaborn as sns
import pickle



# Creating the Title and the Sub-Title
st.write("""
# Money Spent on the Ecommerce Site

""")
st.subheader('Applying Machine Learning Linear Regression using Python!')

# Open and load image
image = Image.open('amazon.jpg')
st.image(image, caption='amazon', use_column_width = True)

# Getting the Data
df = pd.read_csv("Ecommerce Customers")
df = df.iloc[0: , 3:]

# Set sunheader
st.subheader('Data At a Glance')

#Displaying data in table
st.dataframe(df.head())

# Displaying the chart
st.subheader('Correlation Plot of all the Numerical Vairables:')
chart = plt.figure()
sns.heatmap(df.corr(),annot=True)
st.pyplot(chart)


# Get the feature input from Users
def get_user_input():
    Avg_session_length = st.sidebar.slider('Avg. Session Length', 25,40,33)
    Time_on_App = st.sidebar.slider('Time on App', 8,16,9)
    Time_on_website = st.sidebar.slider('Time on Website', 32,40,35)
    Membership_length = st.sidebar.slider('Length of Membership', 0.0,7.0,2.0)

    # Store a dictionary into a variable
    user_data = {'Avg_session_length':Avg_session_length,
                'Time_on_App':Time_on_App,
                'Time_on_website':Time_on_website,
                'Membership_length':Membership_length
                }

    # Transforming data into dataframe
    features = pd.DataFrame(user_data, index = [0])
    return features

# Store the user input into a variable
user_input = get_user_input()

#Setting a subheader and displaying the user input
st.subheader('User Input:')
st.write(user_input)

# Separating X and y
X = df.iloc[:, 0:4].values
Y = df.iloc[: , -1:].values

x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.3,random_state=101)

load_clf = pickle.load(open('EC_clf.pkl', 'rb'))


# Show model metrix
st.subheader('Model Test Accuracy:')
st.write(metrics.explained_variance_score(y_test,load_clf.predict(x_test)))

#Store models predictions in a variable
predictions_of_c = load_clf.predict(user_input)

#Set a subheader ad display result
st.subheader('Amount Spent By Customer: ')
st.write(predictions_of_c)
