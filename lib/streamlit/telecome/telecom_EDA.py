import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV file into a DataFrame
file_path = 'data/telecom.csv'  # Update with the path to your CSV file

# file_path = 'data/telecom_cleaned.csv'  # Update with the path to your CSV file
df = pd.read_csv(file_path)

# Streamlit app
st.title('Exploratory Data Analysis (EDA) App')

# Display basic information about the dataset
st.header("Dataset Information:")
st.write(df.info())

# Display basic statistics of numerical columns
st.header("Descriptive Statistics:")
st.write(df.describe())

# Display the first few rows of the DataFrame
st.header("First Few Rows:")
st.write(df.head())

# Check for missing values
st.header("Missing Values:")
st.write(df.isnull().sum())
def fill_missing_with_mean(df):
    # Iterate through columns and fill missing values with the mean for numerical columns
    for column in df.columns:
        if df[column].dtype in ['float64', 'int64'] and df[column].isnull().any():
            mean_value = df[column].mean()
            df[column].fillna(mean_value, inplace=True)

    return df

# Use the function to fill missing values in your DataFrame
df = fill_missing_with_mean(df)

# Display updated information about missing values
st.header("Updated Missing Values:")
st.write(df.isnull().sum())

def fill_missing_with_mode(df):
    # Iterate through columns and fill missing values with the mode for non-numeric columns
    for column in df.columns:
        if df[column].dtype not in ['float64', 'int64'] and df[column].isnull().any():
            mode_value = df[column].mode().iloc[0]  # Use iloc[0] to get the first mode (handling multimodal distributions)
            df[column].fillna(mode_value, inplace=True)

    return df

# Use the function to fill missing values in your DataFrame
df = fill_missing_with_mode(df)

# Display updated information about missing values
st.header("Updated 2 Missing Values:")
st.write(df.isnull().sum())

df2 = pd.read_csv(file_path)
st.header("Descriptive Statistics:")
st.write(df2.describe())




