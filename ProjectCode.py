# Import necessary libraries
import pandas as pd  # Import pandas for data manipulation and analysis
import numpy as np  # Import numpy for numerical computations
from sklearn.model_selection import train_test_split  # Import train_test_split to split data into training and testing sets
from sklearn.preprocessing import StandardScaler, LabelEncoder  # Import StandardScaler for feature scaling and LabelEncoder for encoding categorical variables
from sklearn.linear_model import LogisticRegression  # Import LogisticRegression for logistic regression modeling
from sklearn.ensemble import RandomForestClassifier  # Import RandomForestClassifier for random forest modeling
from sklearn.metrics import classification_report, roc_auc_score  # Import classification_report and roc_auc_score for model evaluation
from imblearn.over_sampling import SMOTE  # Import SMOTE for handling class imbalance by oversampling the minority class
import matplotlib.pyplot as plt  # Import matplotlib for data visualization
import seaborn as sns  # Import seaborn for enhanced data visualization
import xgboost as xgb

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

#Define the file path
file_path = "/content/drive/My Drive/creditcard.csv"

#Load the dataset
df = pd.read_csv(file_path)

# Convert column names to lowercase and replace spaces with underscores
df.columns = df.columns.str.lower().str.replace(' ', '_')  # Standardize column names for consistency

# Check for missing values
print("Missing values per column:")
print(df.isnull().sum())  # Print the count of missing values for each column in the dataset

# Handle missing values
df.fillna(df.mean(numeric_only=True), inplace=True)  # Fill missing numerical values with the mean of each column
df.fillna(df.mode().iloc[0], inplace=True)  # Fill missing categorical values with the mode (most frequent value) of each column

# Encode categorical variables
label_encoders = {}  # Initialize a dictionary to store label encoders for each categorical column
for col in df.select_dtypes(include=['object']).columns:  # Loop through all categorical columns
    le = LabelEncoder()  # Initialize a LabelEncoder object
    df[col] = le.fit_transform(df[col])  # Encode the categorical column into numerical values
    label_encoders[col] = le  # Store the encoder in the dictionary for potential inverse transformation later

# Scale numerical features
scaler = StandardScaler()  # Initialize a StandardScaler object for feature scaling
numerical_cols = df.select_dtypes(include=['number']).columns  # Select all numerical columns
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])  # Scale numerical features to have mean=0 and std=1


#Removing duplicate data
print("Number of rows before removing duplicates:", len(df))
df = df.drop_duplicates()  # Remove duplicate rows
print("Number of rows after removing duplicates:", len(df))

print("\n Preprocessing Done!\n")

#Display dataset information
print("\n🔹 Dataset Overview:")
print(df.info())

#Display the first few rows
print("Dataset Preview:")
print(df.head())

#Getting basic statistics)
print("\nDataset Summary:")
print(df.describe())

# Check fraud vs. non-fraud transactions
if 'class' in df.columns:  # Check if the 'class' column exists in the DataFrame
    plt.figure(figsize=(6,4))  # Set the figure size for the plot
    ax = sns.countplot(x='class', data=df, palette=['green', 'red'])  # Create a count plot for 'class' with green for non-fraud and red for fraud
    plt.title("Fraud vs. Non-Fraud Transactions")  # Set the title of the plot
    plt.xlabel("Class")  # Label the x-axis
    plt.ylabel("Count")  # Label the y-axis

    # Add the count annotations above each bar
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.0f}',
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center',
                    fontsize=12, color='black',
                    xytext=(0, 5),  # Offset the label a little above the bar
                    textcoords='offset points')

    plt.show()  # Display the plot

    fraud_percentage = df['class'].value_counts(normalize=True) * 100  # Calculate the percentage of fraud and non-fraud transactions
    print("\n🔹 Fraud Percentage:\n", fraud_percentage)  # Print the percentage of fraud and non-fraud transactions

# Print the distribution of the target variable (Class)
print("\nClass Distribution:")
print(df['class'].value_counts())  # Print the count of fraud and non-fraud transactions in the dataset
