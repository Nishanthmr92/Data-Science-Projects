import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import LabelEncoder
import plotly.express as px

# Title
st.title("Titanic Survival Prediction")

# Load Dataset
data = pd.read_csv("SVMtrain.csv")

# Add a button to display the dataset
if st.button("Show Dataset Overview"):
    st.write("Dataset Overview", data.head(50))

# Check if 'PassengerId' and 'Survived' are in the dataset
if 'PassengerId' in data.columns and 'Survived' in data.columns:
    # Label Encoding for categorical features
    label_encoders = {}
    for column in data.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le

    # Select Features
    X = data.drop(['PassengerId', 'Survived'], axis='columns')
    y = data['Survived']

    # Split dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the AdaBoost model
    Ada = AdaBoostClassifier(algorithm='SAMME', n_estimators=100)
    Ada.fit(X_train, y_train)

    # Display model accuracy
    accuracy = Ada.score(X_test, y_test)
    st.write("Model Accuracy:", accuracy)

    st.subheader("Enter Passenger Detail for Prediction")

    # Create input fields for each feature
    user_input = {}
    for feature in X.columns:
        if feature in label_encoders:  # If the feature was label encoded
            options = label_encoders[feature].classes_
            selected_option = st.selectbox(f"{feature}", options)
            user_input[feature] = label_encoders[feature].transform([selected_option])[0]
        else:
            user_input[feature] = st.number_input(f"{feature}", min_value=float(X[feature].min()), max_value=float(X[feature].max()))

    input_df = pd.DataFrame([user_input])

    # Display result
    if st.button("Predict Survival"):
        prediction = Ada.predict(input_df)
        survival_status = "Yes" if prediction[0] == 1 else "No"
        st.write(f"Prediction: Survival or Not? **{survival_status}**")

    # Data Visualization for Gender vs Survival
    data['Survived'] = data['Survived'].replace({1: 'Yes', 0: 'No'})
    fig = px.histogram(
        data,
        x='Sex',
        color='Survived',
        barmode='group',
        labels={'Sex': 'Name Male and Female', 'Survived': 'Survived'},
        title='Number of Male and Female by Survived'
    )
    st.title("Survival Analysis by Gender")
    st.plotly_chart(fig)
else:
    st.write("Dataset does not contain required 'PassengerId' or 'Survived' columns.")


