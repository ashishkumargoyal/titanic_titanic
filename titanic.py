import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Set page configuration
st.set_page_config(
    page_title="Titanic Survival Prediction",
    page_icon="🚢",
    layout="wide"
)

# App title and description
st.title("Titanic Survival Prediction")
st.markdown("""
This app predicts the survival of Titanic passengers using an XGBoost model.
""")

# Function to load the model
@st.cache_resource
def load_model():
    try:
        with open('xg_boost.pkl', 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error("Model file 'xg_boost.pkl' not found. Please make sure it's in the same directory as the app.")
        return None

# Function to load data
@st.cache_data
def load_data():
    # Check if data exists in the repo, otherwise download it
    if os.path.exists('titanic.csv'):
        df = pd.read_csv('titanic.csv')
    else:
        # URL for titanic dataset
        url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
        df = pd.read_csv(url)
        # Save to local file for future use
        df.to_csv('titanic.csv', index=False)
    return df

# Function to preprocess input data
def preprocess_input(
    pclass, sex, age, sibsp, parch, fare, embarked
):
    # Create DataFrame for single passenger
    data = pd.DataFrame({
        'Pclass': [pclass],
        'Sex': [sex],
        'Age': [age],
        'SibSp': [sibsp],
        'Parch': [parch],
        'Fare': [fare],
        'Embarked': [embarked]
    })
    
    # Encode categorical variables
    data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
    data['Embarked'] = data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    
    # Create derived features (similar to those used in training)
    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
    data['IsAlone'] = (data['FamilySize'] == 1).astype(int)
    
    # Ensure all features are available that were used during training
    # Add more feature engineering as needed to match your training pipeline
    
    return data

# Main function
def main():
    # Load the model
    model = load_model()
    
    if model is None:
        st.warning("Please upload 'xg_boost.pkl' to the app directory or train the model first.")
        return
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Prediction", "Dataset Overview"])
    
    # Load the dataset for exploration
    df = load_data()
    
    if page == "Dataset Overview":
        st.header("Titanic Dataset Overview")
        
        # Show raw data
        st.subheader("Raw Data")
        st.write(df.head())
        
        # Basic statistics
        st.subheader("Data Statistics")
        st.write(df.describe())
        
        # Visualizations
        st.subheader("Data Visualizations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Survival by sex
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.countplot(x='Sex', hue='Survived', data=df, ax=ax)
            ax.set_title('Survival by Sex')
            st.pyplot(fig)
            
            # Survival by class
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.countplot(x='Pclass', hue='Survived', data=df, ax=ax)
            ax.set_title('Survival by Class')
            st.pyplot(fig)
        
        with col2:
            # Age distribution
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.histplot(df['Age'].dropna(), kde=True, bins=20, ax=ax)
            ax.set_title('Age Distribution')
            st.pyplot(fig)
            
            # Fare distribution
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.histplot(df['Fare'].dropna(), kde=True, bins=20, ax=ax)
            ax.set_title('Fare Distribution')
            st.pyplot(fig)
    
    elif page == "Prediction":
        st.header("Passenger Survival Prediction")
        
        st.subheader("Enter Passenger Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            pclass = st.selectbox("Passenger Class", [1, 2, 3], index=2)
            sex = st.radio("Sex", ["male", "female"])
            age = st.slider("Age", 0.5, 80.0, 30.0)
            sibsp = st.slider("Number of Siblings/Spouses", 0, 8, 0)
        
        with col2:
            parch = st.slider("Number of Parents/Children", 0, 6, 0)
            fare = st.slider("Fare (£)", 0.0, 512.0, 32.0, step=1.0)
            embarked = st.selectbox("Port of Embarkation", ["S", "C", "Q"], index=0)
        
        # When the button is clicked
        if st.button("Predict Survival"):
            # Preprocess the input
            input_data = preprocess_input(
                pclass, sex, age, sibsp, parch, fare, embarked
            )
            
            # Make prediction
            try:
                prediction = model.predict(input_data)
                probability = model.predict_proba(input_data)
                
                # Display prediction
                st.subheader("Prediction Result")
                
                if prediction[0] == 1:
                    st.success(f"Survival Prediction: SURVIVED")
                    st.info(f"Probability of survival: {probability[0][1]:.2%}")
                else:
                    st.error(f"Survival Prediction: DID NOT SURVIVE")
                    st.info(f"Probability of not surviving: {probability[0][0]:.2%}")
                
                # Feature importance visualization
                if hasattr(model, 'feature_importances_'):
                    st.subheader("Feature Importance")
                    
                    # Create feature importance dataframe
                    feature_importance = pd.DataFrame({
                        'Feature': input_data.columns,
                        'Importance': model.feature_importances_
                    }).sort_values('Importance', ascending=False)
                    
                    # Plot
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.barplot(x='Importance', y='Feature', data=feature_importance, ax=ax)
                    ax.set_title('Feature Importance')
                    st.pyplot(fig)

                # Include explanation of results
                st.subheader("What Influenced This Prediction?")
                
                # Explain the key factors
                if sex == "female" and pclass == 1:
                    st.write("First-class female passengers had among the highest survival rates on the Titanic.")
                elif sex == "female":
                    st.write("Female passengers generally had higher survival rates due to 'women and children first' policy.")
                elif age < 12:
                    st.write("Children were prioritized for rescue, improving survival chances.")
                elif pclass == 3 and sex == "male":
                    st.write("Third-class male passengers had among the lowest survival rates on the Titanic.")
                
                # Additional factors based on other inputs
                if input_data['IsAlone'].values[0] == 1:
                    st.write("Passengers traveling alone often had different survival outcomes than those with family.")
                
                if fare > 100:
                    st.write("Passengers with expensive tickets were often in higher classes with better access to lifeboats.")
            
            except Exception as e:
                st.error(f"Error making prediction: {e}")
                st.write("The model might be expecting different features than what was provided. Please check that the model was trained with the same features used for prediction.")

# Add instructions for deployment
st.sidebar.markdown("""
### Deployment Instructions
1. Upload these files to GitHub:
   - app.py
   - xg_boost.pkl
   - requirements.txt
2. Connect your GitHub repository to Streamlit Cloud
3. Deploy the app
""")

# Add information about the model
st.sidebar.markdown("""
### About the Model
This app uses an XGBoost classifier trained on the Titanic dataset to predict passenger survival.
""")

# Run the app
if __name__ == "__main__":
    main()