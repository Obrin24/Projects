import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.calibration import LabelEncoder
from src.training import train_logistic_regression, train_svm, train_fnn
from src.evaluate import evaluate_logistic_regression, evaluate_svm, evaluate_fnn

# Load data
def load_data():
    # Load your data
    Symptoms_df = pd.read_csv(r'C:\Users\gabeb\Desktop\programming\Projects\Medical\Disease Prediction\DiseaseAndSymptoms.csv')
        
    # Replace NaN values with 'None'
    Symptoms_df = Symptoms_df.fillna('None')
    
    # Encode categorical variables
    label_encoders = {}
    for col in Symptoms_df.columns:
        if Symptoms_df[col].dtype == 'object':
            label_encoders[col] = LabelEncoder()
            Symptoms_df[col] = label_encoders[col].fit_transform(Symptoms_df[col])
    
    # Split features and target
    X = Symptoms_df.drop('Disease', axis=1)
    y = Symptoms_df['Disease']
    
    return X, y

# Split data into train and test sets
def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test

# Main function
def main():
    st.title("Disease Prediction")
    
    # Load data
    X, y = load_data()    
    
    # Sidebar options
    model_choice = st.sidebar.radio("Select Model", ("Logistic Regression", "SVM", "Feedforward Neural Network"))
    evaluate_model = st.sidebar.checkbox("Evaluate Model")
    predict_symptoms = st.sidebar.checkbox("Predict Disease")

    if evaluate_model:
        st.sidebar.write("Evaluation in progress...")

        if model_choice == "Logistic Regression":
            X_train, X_test, y_train, y_test = split_data(X, y)
            model = train_logistic_regression(X_train, y_train)
            accuracy = evaluate_logistic_regression(model, X_test, y_test)
            st.sidebar.write("Accuracy:", accuracy)

        elif model_choice == "SVM":
            X_train, X_test, y_train, y_test = split_data(X, y)
            model = train_svm(X_train, y_train)
            accuracy = evaluate_svm(model, X_test, y_test)
            st.sidebar.write("Accuracy:", accuracy)

        elif model_choice == "Feedforward Neural Network":
            X_train, X_test, y_train, y_test = split_data(X, y)
            input_size = X_train.shape[1]
            model = train_fnn(X_train, y_train, input_size)
            accuracy = evaluate_fnn(model, X_test, y_test)
            st.sidebar.write("Accuracy:", accuracy)

    if predict_symptoms:
        st.sidebar.write("Enter your symptoms:")
        symptoms = {}
        for col in X.columns:
            if col != 'Disease':
                symptoms[col] = st.sidebar.selectbox(f"{col}:", X[col].unique())

        symptoms_df = pd.DataFrame([symptoms])
        predicted_disease = model.predict(symptoms_df)[0]

        st.write("Predicted Disease:", predicted_disease)

        # Compare with actual dataset
        actual_disease = y[X[X.eq(symptoms_df.iloc[0]).all(1)].index[0]]
        st.write("Actual Disease in Dataset:", actual_disease)

if __name__ == "__main__":
    main()
