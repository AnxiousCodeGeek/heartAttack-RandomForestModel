# %%
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
# from scikeras.wrappers import KerasClassifier

import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, Conv2D, MaxPooling2D, GlobalAveragePooling2D

# %%
df = pd.read_csv(r'D:\internee.pk\TSK-000-186\heartattack_processed_data.csv')

df.head(-1)

# %%
df = df.drop(columns=['Troponin', 'CK-MB'])

# %%
df = df.drop(columns='CK-MB_normalized')

# %%
# Split the dataset into features (X) and target variable (y)
X = df[['Age', 'Gender', 'Heart rate', 'Systolic blood pressure', 'Diastolic blood pressure']]
y = df['Result']

# %%
# Train a random forest classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

# %%
# Get feature importance scores
importances = rf.feature_importances_

# %%
# Plot feature importances
plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.bar(X.columns, importances)
plt.xlabel("Feature")
plt.ylabel("Importance Score")
plt.xticks(rotation=45)
plt.show()


# %%
# Create a DataFrame to store feature importance scores
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})

# Sort features by importance scores in descending order
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Set threshold for selecting relevant features (e.g., importance score > 0.05)
threshold = 0.05

# Identify irrelevant features based on importance scores
irrelevant_features = feature_importance_df[feature_importance_df['Importance'] < threshold]['Feature']

# Remove irrelevant features from DataFrame
filtered_df = df.drop(columns=irrelevant_features)

# %%
filtered_df.head(-1)

# %% [markdown]
# we will not use filtered_df data because gender is also an important feature

# %%

# Calculate z-scores for each column
z_scores = (df - df.mean()) / df.std()

# Set threshold for outlier detection (e.g., z-score > 3 or < -3)
threshold = 3

# Identify outliers based on z-scores
outliers = (np.abs(z_scores) > threshold).any(axis=1)

# Remove outliers from DataFrame
cleaned_df = df[~outliers]


# %%
cleaned_df.reset_index(drop=True, inplace=True)
cleaned_df.head(-1)

# %% [markdown]
# **Generating and adding random height and weight data to the dataset**

# %%

# Set seed for reproducibility
np.random.seed(0)

# Generate random weight data in kg (assuming a normal distribution with mean 70 kg and standard deviation 10 kg)
random_weight = np.random.normal(loc=70, scale=10, size=len(cleaned_df)-1)

# Generate random height data in meters (assuming a normal distribution with mean 1.7 meters and standard deviation 0.1 meters)
random_height = np.random.normal(loc=1.7, scale=0.1, size=len(cleaned_df)-1)

# Insert NaN value at index 0 for weight and height arrays
random_weight = np.insert(random_weight, 0, np.nan)
random_height = np.insert(random_height, 0, np.nan)

# Add the random weight and height data to your dataset
cleaned_df['weight'] = random_weight
cleaned_df['height'] = random_height
cleaned_df

# %% [markdown]
# Removing the null value row

# %%
cleaned_df = cleaned_df.dropna(axis=0)
cleaned_df.reset_index(drop=True, inplace=True)
cleaned_df.head(-1)

# %% [markdown]
# ## Feature Creation and Engineering

# %% [markdown]
# **Body Mass Index (BMI)**

# %%
# Calculate BMI (Body Mass Index)
cleaned_df['BMI'] = cleaned_df['weight'] / (cleaned_df['height'] ** 2)
cleaned_df

# %% [markdown]
# **Pulse Pressure**

# %%
# Calculate Pulse Pressure
cleaned_df['pulse_pressure'] = cleaned_df['Systolic blood pressure'] - cleaned_df['Diastolic blood pressure']
cleaned_df

# %% [markdown]
# **Categorize Condition based on Blood pressure**

# %%
# Categorize Blood Pressure
def categorize_blood_pressure(row):
    if row['Systolic blood pressure'] < 120 and row['Diastolic blood pressure'] < 80:
        return 'Normal'
    elif 120 <= row['Systolic blood pressure'] < 130 or 80 <= row['Diastolic blood pressure'] < 85:
        return 'Pre-Hypertension'
    else:
        return 'Hypertension'

cleaned_df['BP_based_Condition'] = cleaned_df.apply(categorize_blood_pressure, axis=1)
cleaned_df

# %% [markdown]
# ## EDA

# %%

fig = plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
sns.distplot(cleaned_df['pulse_pressure'])
plt.title('Pulse Pressure (mmHg) Distribution')
plt.axvline(x=60, color='r', linestyle='--', label='High pulse pressure: over 60 mmHg')
plt.ylabel('')

plt.subplot(1, 2, 2)
sns.distplot(cleaned_df['BMI'])
plt.title('Distribution of BMI')
# plt.xlim(10,50)
# plt.axvline(x=60, color='r', linestyle='--', label='High pulse pressure: over 60 mmHg')
plt.ylabel('')
plt.legend()

# %%


cleaned_df = cleaned_df.dropna(subset=['Gender','pulse_pressure', 'BMI'])
fig = plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
sns.histplot(data=cleaned_df, x='pulse_pressure', hue='Gender', element='step', palette=['Red', 'Blue'], kde=True)
plt.title('Pulse Pressure (mmHg) Distribution for Each Gender')
plt.axvline(x=60, color='r', linestyle='--', label='High pulse pressure: over 60 mmHg')
plt.ylabel('')


plt.subplot(1, 2, 2)
sns.histplot(data=cleaned_df, x='BMI', hue='Gender', element='step', palette=['Red', 'Blue'], kde=True)
plt.title('Distribution of BMI for Each Gender')
# plt.xlim(10,50)
# plt.axvline(x=60, color='r', linestyle='--', label='High pulse pressure: over 60 mmHg')
plt.ylabel('')



# %%
fig = plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
positive = cleaned_df[cleaned_df['Result']==0]['pulse_pressure']
negative = cleaned_df[cleaned_df['Result']==1]['pulse_pressure']
sns.histplot(negative, color='Green')
sns.histplot(positive, color='Red')
plt.title('Pulse Pressure (mmHg) Distribution for Heartattack results')
plt.axvline(x=60, color='r', linestyle='--', label='High pulse pressure: over 60 mmHg')
plt.ylabel('')


plt.subplot(1, 2, 2)
positive = cleaned_df[cleaned_df['Result']==0]['BMI']
negative = cleaned_df[cleaned_df['Result']==1]['BMI']
sns.histplot(negative, color='Green')
sns.histplot(positive, color='Red')
plt.title('Distribution of BMI for Heartattack results')
# plt.xlim(10,50)
# plt.axvline(x=60, color='r', linestyle='--', label='High pulse pressure: over 60 mmHg')
plt.ylabel('')


# %%
cleaned_df = cleaned_df.dropna(subset=['Age','Gender','pulse_pressure', 'BMI'])
fig = plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
sns.scatterplot(data=cleaned_df, x='Age', y='pulse_pressure', hue=cleaned_df['Result'].apply(lambda x: 'Negative' if x == 0 else 'Positive'), palette='Set1')
plt.title('Age vs Pulse Pressure')
plt.axvline(x=60, color='r', linestyle='--', label='High pulse pressure: over 60 mmHg')
plt.ylabel('Pulse Pressure (mmHg)')
plt.xlabel('Age (years)')


# %% [markdown]
# # Checking sample size of ```Gender``` and ```Age```

# %%
# checking Age label distribution in dataset

print(cleaned_df["Age"].value_counts())
cleaned_df['Age'].hist(figsize=(5,5))

# %%
print(cleaned_df["Gender"].value_counts())
cleaned_df['Gender'].hist(figsize=(5,5))

# %% [markdown]
# For ```Gender``` label, the sample distribution for male and female is not equal so we resample them.

# %%
from sklearn.utils import resample

# Separate male and female data
male_data = cleaned_df[cleaned_df['Gender'] == 1]
female_data = cleaned_df[cleaned_df['Gender'] == 0]

# Resample female data to match the number of male data
female_resampled = resample(female_data, replace=True, n_samples=len(male_data), random_state=42)

# Combine resampled female data with male data
cleaned_df = pd.concat([male_data, female_resampled])
print(cleaned_df["Gender"].value_counts())
cleaned_df['Gender'].hist(figsize=(5,5))

# %% [markdown]
# Also it is observed that the ```Age``` groups are unevenly distributed in the data, with most the data derived from older age groups than the younger. To solve this we can either remove young people data as it is less or we can resample our data. 

# %%
# Define age groups
age_groups = [0, 20, 30, 40, 50, 60, 70, 80, 90, 100]

# Initialize a list to store resampled data for each age group
resampled_data = []

# Loop through each age group
for i in range(len(age_groups) - 1):
    # Extract data for the current age group
    age_group_data = cleaned_df[(cleaned_df['Age'] >= age_groups[i]) & (cleaned_df['Age'] < age_groups[i+1])]
    
    # Resample the data to have a minimum of 50 samples per age group
    resampled_age_group = resample(age_group_data, replace=True, n_samples=50, random_state=42)
    
    # Add the resampled data to the list
    resampled_data.append(resampled_age_group)

# Combine the resampled data for all age groups
cleaned_df = pd.concat(resampled_data)

# %%
print(cleaned_df["Age"].value_counts())
cleaned_df['Age'].hist(figsize=(5,5))

# %% [markdown]
# ## Data Transformation:

# %%
# Perform one-hot encoding for the 'BP_based_condition' column
cleaned_df_encoded = pd.get_dummies(cleaned_df, columns=['BP_based_Condition'])

# Display the first few rows of the encoded dataframe
cleaned_df_encoded.reset_index(drop=True, inplace=True)
print(cleaned_df_encoded.head())

# %% [markdown]
# ## Model Selection:

# %% [markdown]
# ### For the heart attack, the problem type is classification, as the target variable 'Result' is categorical (indicating the presence or absence of a heart attack). 
# 
# The appropiate machine learning algorithm that i would consider for this problem would be either Random Forest classifier or Neural Networks.

# %% [markdown]
# The choice of algorithm depends on the specific problem and the trade-offs between model complexity, interpretability, and computational efficiency. For example, if interpretability and computational efficiency is important, Random Forest may be a good choice. If model complexity is important, Neural Networks may be a good choice. It is important to evaluate and compare the performance of different algorithms based on appropriate evaluation metrics and to choose the algorithm that best balances these factors.

# %% [markdown]
# ### Splitting data into ```test``` and ```train``` and building model to train

# %%
X = cleaned_df_encoded.drop(columns=['Result'])
y = cleaned_df_encoded['Result']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %%
from keras.layers import Flatten

# Create a Sequential model
model = Sequential()

# Add an input layer with 11 nodes (one for each feature)
model.add(Dense(11, activation='relu', input_shape=(X_train.shape[1],)))

# Add a hidden layer with 6 nodes
model.add(Dense(6, activation='relu'))
model.add(Flatten())

# Add an output layer with 1 node (for binary classification)
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# %%
history = model.fit(X_train, y_train, 
                    epochs=500, 
                    batch_size=10, 
                    validation_data=(X_test, y_test))

# %%
# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test accuracy: {accuracy}')

# %%
# Plot the accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()

# Plot the loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()



