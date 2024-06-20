import sqlite3
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib
import streamlit as st

# Load the CSV data into a DataFrame
heart_csv = 'heart.csv'
df = pd.read_csv(heart_csv, delimiter=';')
print(df.head())

# Create a SQLite database and insert data
conn = sqlite3.connect('heartData.db')
cursor = conn.cursor()
cursor.execute('''
CREATE TABLE IF NOT EXISTS heartInfo (
    age INTEGER,
    sex INTEGER,
    cp INTEGER,
    trestbps INTEGER,
    chol INTEGER,
    fbs INTEGER,
    restecg INTEGER,
    thalach INTEGER,
    exang INTEGER,
    oldpeak REAL,
    slope INTEGER,
    ca INTEGER,
    thal INTEGER,
    target INTEGER
)
''')
conn.commit()

try:
    df.to_sql('heartInfo', conn, if_exists='append', index=False)
    print("Data has been successfully inserted")
except Exception as e:
    print("Data has not been successfully inserted: ", e)

conn.commit()
conn.close()

def query_database(db_path):
    conn = sqlite3.connect(db_path)
    query = "SELECT age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, target FROM heartInfo"
    df = pd.read_sql_query(query, conn)
    print(df.head())
    conn.close()
    return df

# Visualize categorical variables
categorical_vars = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
plt.figure(figsize=(20, 15))
for i, var in enumerate(categorical_vars, 1):
    plt.subplot(3, 3, i)
    sns.countplot(data=df, x=var, hue='target')
    plt.title(f'Distribution of {var} by Target')
    plt.xlabel(var)
    plt.ylabel('Count')
plt.tight_layout()
plt.show()

# Visualize non-categorical variables
noncategorical_vars = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
plt.figure(figsize=(20, 15))
for i, var in enumerate(noncategorical_vars, 1):
    plt.subplot(3, 3, i)
    sns.histplot(data=df, x=var, hue='target', kde=True)
    plt.title(f'Distribution of {var} by Target')
    plt.xlabel(var)
    plt.ylabel('Count')
plt.tight_layout()
plt.show()

# Data Preprocessing and Model Training
training_1 = df.drop('target', axis=1)
training_2 = df['target']
x_training_1, x_testing_1, y_training_2, y_testing_2 = train_test_split(training_1, training_2, test_size=0.2, random_state=32)

noncategorical_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']

noncategorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')), 
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('noncat', noncategorical_transformer, noncategorical_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

x_training_1 = preprocessor.fit_transform(x_training_1)
x_testing_1 = preprocessor.transform(x_testing_1)

print(f"x_training_1 shape: {x_training_1.shape}")
print(f"x_testing_1 shape: {x_testing_1.shape}")
print(f"y_training_2 shape: {y_training_2.shape}")
print(f"y_testing_2 shape: {y_testing_2.shape}")

models = {
    'Logistic Regression': LogisticRegression(random_state=32),
    'Random Forest': RandomForestClassifier(random_state=32),
    'Support Vector Machine': SVC(probability=True, random_state=32)
}

best_model = None
best_auc = 0

for model_name, model in models.items():
    model.fit(x_training_1, y_training_2)
    y_pred = model.predict(x_testing_1)
    y_proba = model.predict_proba(x_testing_1)[:, 1]

    print(f"Model: {model_name}")
    print(classification_report(y_testing_2, y_pred))

    auc = roc_auc_score(y_testing_2, y_proba)
    print('AUC-ROC:', auc)
    print('-' * 50)
    
    if auc > best_auc:
        best_model = model
        best_auc = auc

print(f"Best Model: {best_model}")
print(f"Best AUC-ROC: {best_auc}")

# Save the best model and preprocessor
joblib.dump(best_model, 'best_model.pkl')
joblib.dump(preprocessor, 'preprocessor.pkl')

# Load the saved model and preprocessor in the Streamlit app
model = joblib.load('best_model.pkl')
preprocessor = joblib.load('preprocessor.pkl')

def get_user_input():
    age = st.number_input('Age', min_value=1, max_value=120, value=25)
    sex = st.selectbox('Sex', [0, 1])  # 0 for female, 1 for male
    cp = st.selectbox('Chest Pain Type (cp)', [0, 1, 2, 3])
    trestbps = st.number_input('Resting Blood Pressure (trestbps)', min_value=0, max_value=300, value=120)
    chol = st.number_input('Serum Cholesterol in mg/dl (chol)', min_value=0, max_value=600, value=200)
    fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl (fbs)', [0, 1])
    restecg = st.selectbox('Resting ECG (restecg)', [0, 1, 2])
    thalach = st.number_input('Maximum Heart Rate Achieved (thalach)', min_value=0, max_value=220, value=150)
    exang = st.selectbox('Exercise Induced Angina (exang)', [0, 1])
    oldpeak = st.number_input('ST Depression Induced by Exercise (oldpeak)', min_value=0.0, max_value=10.0, value=1.0)
    slope = st.selectbox('Slope of the Peak Exercise ST Segment (slope)', [0, 1, 2])
    ca = st.selectbox('Number of Major Vessels Colored by Fluoroscopy (ca)', [0, 1, 2, 3, 4])
    thal = st.selectbox('Thalassemia (thal)', [0, 1, 2, 3])
    
    user_data = {
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal
    }
    
    features = pd.DataFrame(user_data, index=[0])
    return features

def main():
    st.title('Heart Disease Prediction App')
    st.write('Enter the details of the patient to predict the likelihood of heart disease.')

    input_df = get_user_input()

    st.subheader('Patient Data')
    st.write(input_df)

    preprocessed_input = preprocessor.transform(input_df)

    prediction = model.predict(preprocessed_input)
    prediction_proba = model.predict_proba(preprocessed_input)[:, 1]

    st.subheader('Prediction')
    heart_disease_risk = 'High' if prediction[0] == 1 else 'Low'
    st.write(f'The risk of heart disease is: {heart_disease_risk}')
    st.write(f'Probability of having heart disease: {prediction_proba[0]:.2f}')

if __name__ == '__main__':
    main()
