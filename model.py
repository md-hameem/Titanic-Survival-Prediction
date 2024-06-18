import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

# Load dataset (assuming it's in CSV format)
data = pd.read_csv('titanic.csv')

# Preprocess the data
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
data = pd.get_dummies(data, columns=['Embarked'], drop_first=True)

# Check the columns to ensure all required columns are present
print(data.columns)

# Define feature matrix and target vector
X = data[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex', 'Embarked_Q', 'Embarked_S']].copy()
y = data['Survived']

# Handle missing values
X.fillna(X.mean(), inplace=True)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Save the model
with open('titanic_v3.pkl', 'wb') as file:
    pickle.dump(model, file)

# Save the scaler (optional, if you want to use the same scaling in the app)
with open('scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)
