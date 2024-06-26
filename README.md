# Titanic Survival Prediction

This project predicts the survival chances of passengers aboard the Titanic using a machine learning model. The project is implemented using Python and Streamlit, and it leverages historical data from the Titanic disaster.

## Project Overview

The Titanic Survival Prediction project aims to provide insights into the factors that influenced survival during the Titanic disaster. By using machine learning techniques, the project predicts whether a passenger would have survived based on various features such as age, sex, passenger class, and more.

## Features

- Predict the survival chances of a passenger.
- Interactive web application built with Streamlit.
- Visual insights and historical facts about the Titanic disaster.

## Dataset

The dataset used for training the model is sourced from the Titanic dataset available on Kaggle. The dataset includes information about the passengers such as:
- PassengerId
- Survived
- Pclass
- Name
- Sex
- Age
- SibSp
- Parch
- Ticket
- Fare
- Cabin
- Embarked

## Installation

To run this project locally, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/md-hameem/Titanic-Survival-Prediction.git
    ```
2. Navigate to the project directory:
    ```bash
    cd Titanic-Survival-Prediction
    ```
3. Create a virtual environment:
    ```bash
    python -m venv venv
    ```
4. Activate the virtual environment:
    - On Windows:
        ```bash
        venv\Scripts\activate
        ```
    - On macOS and Linux:
        ```bash
        source venv/bin/activate
        ```
5. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To run the Streamlit web application, use the following command:
```bash
streamlit run TitanicWeb.py
```

This will start the web application, and you can interact with it through your web browser.

## Project Structure

- `Colab-Jupyter/` : Jupyter notebooks for data exploration and model training, i used Google Colab for better experience.
- `dataset/` : Contains the Titanic dataset.
- `images/` : Images used in the web application.
- `model.py` : Script for training the machine learning model.
- `TitanicWeb.py` : Streamlit application script.
- `scaler.pkl` : Scaler object for data normalization.
- `titanic_v0.pkl` : Trained machine learning model.
- `requirements.txt` : List of required Python packages.
- `README.md` : Project documentation.

## How It Works

1. **Data Collection**: The project uses historical data from the Titanic disaster, which includes various features like age, gender, ticket class, number of siblings/spouses aboard, number of parents/children aboard, fare, and boarding location.

2. **Data Preprocessing**: The data undergoes preprocessing to handle missing values and categorical variables. For example, gender is converted to numerical values (0 for male, 1 for female), and boarding locations are encoded as one-hot vectors.

3. **Feature Scaling**: The features are scaled to ensure that the model can learn effectively. This scaling is done using the `StandardScaler` from scikit-learn.

4. **Model Training**: A logistic regression model is trained on the processed dataset. This model learns the patterns and relationships between the features and the survival outcome.

5. **Making Predictions**: The trained model is used to make predictions on new data. In this app, you can input your own details to see whether you would have survived the Titanic disaster.

6. **Displaying Results**: The app provides a probability of survival and a prediction on whether you would have survived. It also includes additional insights and fun facts about the Titanic disaster.

## Author

This project is developed by Mohammad Hamim.

## License

This project is licensed under the MIT License.
