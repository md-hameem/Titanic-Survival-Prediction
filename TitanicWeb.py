import streamlit as st
import pandas as pd
import pickle

# Load the model and scaler
model = pickle.load(open('titanic_v3.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

def main():
    st.title("Titanic Survival Prediction")
    st.image('titanic_sinking.jpg', caption="Sinking of 'RMS Titanic': 15 April 1912 in North Atlantic Ocean", use_column_width=True)
    st.write("## Would you have survived From Titanic Disaster?")

    st.title("----- Check Your Survival Chances -----")
    st.write("""
        ### How Our Project Works:
        This project aims to predict the likelihood of a passenger surviving the Titanic disaster using machine learning. Here's a step-by-step explanation of the process:

        1. **Data Collection**:
            - We use historical data from the Titanic disaster, which includes various features like age, gender, ticket class, number of siblings/spouses aboard, number of parents/children aboard, fare, and boarding location.

        2. **Data Preprocessing**:
            - The data undergoes preprocessing to handle missing values and categorical variables. For example, gender is converted to numerical values (0 for male, 1 for female), and boarding locations are encoded as one-hot vectors.

        3. **Feature Scaling**:
            - The features are scaled to ensure that the model can learn effectively. This scaling is done using the `StandardScaler` from scikit-learn.

        4. **Model Training**:
            - A logistic regression model is trained on the processed dataset. This model learns the patterns and relationships between the features and the survival outcome.

        5. **Making Predictions**:
            - The trained model is used to make predictions on new data. In this app, you can input your own details to see whether you would have survived the Titanic disaster.

        6. **Displaying Results**:
            - The app provides a probability of survival and a prediction on whether you would have survived. It also includes additional insights and fun facts about the Titanic disaster.
    """)

    age = st.slider("Enter Age:", 1, 75, 30)
    fare = st.slider("Fare (in 1912 $):", 15, 500, 40)
    SibSp = st.selectbox("How many Siblings or spouses are travelling with you?", [0, 1, 2, 3, 4, 5, 6, 7, 8])
    Parch = st.selectbox("How many Parents or children are travelling with you?", [0, 1, 2, 3, 4, 5, 6, 7, 8])
    sex = st.selectbox("Select Gender:", ["Male", "Female"])
    Pclass = st.selectbox("Select Passenger-Class:", [1, 2, 3])
    boarded_location = st.selectbox("Boarded Location:", ["Southampton", "Cherbourg", "Queenstown"])

    Sex = 1 if sex == "Female" else 0
    Embarked_Q = 1 if boarded_location == "Queenstown" else 0
    Embarked_S = 1 if boarded_location == "Southampton" else 0

    data = {"Pclass": Pclass, "Age": age, "SibSp": SibSp, "Parch": Parch, "Fare": fare, "Sex": Sex, "Embarked_Q": Embarked_Q, "Embarked_S": Embarked_S}
    df = pd.DataFrame(data, index=[0])

    # Apply the same scaling as during training
    df_scaled = scaler.transform(df)

    return df_scaled

data = main()

if st.button("Predict"):
    result = model.predict(data)
    proba = model.predict_proba(data)
    if result[0] == 1:
        st.write("***Congratulations !!!....*** **You probably would have made it!**")
        st.image('lifeboat.jfif')
        st.write("**Survival Probability Chances :** 'NO': {}%  'YES': {}% ".format(round((proba[0,0])*100,2), round((proba[0,1])*100,2)))
    else:
        st.write("***Better Luck Next time !!!!...*** **You're probably ended up like 'Jack'**")
        st.image('Rip.jfif')
        st.write("**Survival Probability Chances :** 'NO': {}%  'YES': {}% ".format(round((proba[0,0])*100,2), round((proba[0,1])*100,2)))
if st.button("Author"):
    st.write("Mohammad Hamim - 202280090114")
