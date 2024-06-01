# Heart Attack Prediction: Random Forest Model

## Introduction
This project aims to predict the likelihood of a heart attack based on various health indicators using machine learning techniques. The dataset contains features such as age, gender, heart rate, blood pressure, weight, height, and other derived features like BMI and pulse pressure. The goal is to build a robust model to assist in identifying high-risk individuals.

## Requirements
 - ```pandas```
 - ```scikit-learn```
 - ```matplotlib```
 - ```seaborn```
 - ```tensorflow```
 - ```keras```
 - ```numpy```
 - ```imbalanced-learn```

## Working
 - The dataset is split into training and testing sets, and a Random Forest classifier is trained.
```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

X = cleaned_df_encoded.drop(columns=['Result'])
y = cleaned_df_encoded['Result']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
```
 - The model is evaluated using accuracy score, confusion matrix, and classification report.
```python
y_pred = rf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
```

### Hyperparameter Tuning: 
Hyperparameter tuning is the process of finding the optimal combination of hyperparameters for a machine learning model that results in the best performance on a given dataset. The ```param_grid``` dictionary defines the hyperparameters to be tuned, along with their possible values. In this case, the hyperparameters are:
 - ```n_estimators```: The number of decision trees in the random forest classifier. Possible values are 50, 100, and 200.
 - ```max_depth```: The maximum depth of each decision tree. Possible values are None (no limit), 5, and 10.

The ```GridSearchCV``` object is created, passing in the random forest classifier (```rf```) and the hyperparameter grid (```param_grid```).  The ```fit``` method is called on the ```GridSearchCV``` object, passing in the training data (```X_train``` and ```y_train```). The grid search algorithm will train and evaluate the model for each combination of hyperparameters, and store the results.
```python
param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [None, 5, 10]}
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
```
The ```best_params_``` returns the combination of hyperparameters that resulted in the best performance (highest accuracy).
```python
print("Best Parameters:", grid_search.best_params_)
```

The ```best_score_``` attribute returns the best accuracy score achieved by the model with the best hyperparameters.
```python
print("Best Score:", grid_search.best_score_)
```

A new random forest classifier is created with the best hyperparameters, and trained on the entire training dataset.
```python
best_rf = RandomForestClassifier(**grid_search.best_params_)
best_rf.fit(X_train, y_train)
```

The output of this code will be:

 - The best combination of hyperparameters (```best_params_```)
 - The best accuracy score achieved by the model with the best hyperparameters (```best_score_```)
 - The accuracy score of the best model on the test dataset (```accuracy_best```)
```python
y_pred_best = best_rf.predict(X_test)
accuracy_best = accuracy_score(y_test, y_pred_best)
print("Best Accuracy:", accuracy_best)
```

The project was carried out in two parts :
 - Building model and predictions based on the original dataset ```cleaned_df_encoded()```

```
Accuracy: 0.6138996138996139
Confusion Matrix:
[[ 34  66]
 [ 34 125]]
Classification Report:
              precision    recall  f1-score   support

           0       0.50      0.34      0.40       100
           1       0.65      0.79      0.71       159

    accuracy                           0.61       259
   macro avg       0.58      0.56      0.56       259
weighted avg       0.59      0.61      0.59       259

Best Parameters: {'max_depth': 5, 'n_estimators': 50}
Best Score: 0.6450494817316261
Best Accuracy: 0.637065637065637
```

 - **Resampling** the ```Age``` labels because most the data was deviated towards older people rather than younger , then again splitting data and building model.
```python
# Define the minority and majority classes based on the Age column
minority_class = cleaned_df[cleaned_df['Age'] < cleaned_df['Age'].quantile(0.25)]
majority_class = cleaned_df[cleaned_df['Age'] >= cleaned_df['Age'].quantile(0.25)]

# Define the target variables for the minority and majority classes
minority_target = np.ones(len(minority_class))
majority_target = np.zeros(len(majority_class))

# Combine the data and target variables
combined_df = pd.concat([minority_class, majority_class])
combined_target = np.concatenate([minority_target, majority_target])

# Define the oversampling and undersampling objects
ros = RandomOverSampler(sampling_strategy='auto')
rus = RandomUnderSampler(sampling_strategy='auto')

# First, upsample the minority class
X_resampled, y_resampled = ros.fit_resample(combined_df, combined_target)

# Then, downsample the majority class
X_resampled, y_resampled = rus.fit_resample(X_resampled, y_resampled)

# Combine the resampled data into a DataFrame
resampled_df = pd.DataFrame(X_resampled, columns=cleaned_df.columns)
resampled_df['Target'] = y_resampled

# Shuffle the data
resampled_df = resampled_df.sample(frac=1).reset_index(drop=True)

# Display the first 10 rows of the resampled data
print(resampled_df.head())
```

This showed us difference in the accuracy based on number of samples of the data.
```
Accuracy: 0.8261964735516373
Confusion Matrix:
[[120  57]
 [ 12 208]]
Classification Report:
              precision    recall  f1-score   support

           0       0.91      0.68      0.78       177
           1       0.78      0.95      0.86       220

    accuracy                           0.83       397
   macro avg       0.85      0.81      0.82       397
weighted avg       0.84      0.83      0.82       397

Best Parameters: {'max_depth': 10, 'n_estimators': 200}
Best Score: 0.7987381703470031
Best Accuracy: 0.8211586901763224
```

## Predictions
We created an example data of a patient to test our predictions
```python
patient_data = pd.DataFrame({
    'Age': [20],
    'Gender': [1],  # 1 for male, 0 for female
    'Heart rate': [78],
    'Systolic blood pressure': [125],
    'Diastolic blood pressure': [87],
    'weight': [40],
    'height': [1.7],
    'BMI': [24.5],
    'pulse_pressure': [40],
    'BP_based_Condition': [0]  # 0 for Normal, 1 for Pre-Hypertension, 2 for Hypertension
})
```
Preprocessing the data for predictions
```python
# Fit the LabelEncoder on all possible labels
le.fit(patient_data['BP_based_Condition'].unique())

# Encode the categorical value
patient_data['BP_based_Condition'] = le.transform(patient_data['BP_based_Condition'])

# Make predictions
prediction = best_rf.predict(patient_data)

# Print the prediction result
if prediction[0] == 1:
    print("The patient is likely to have a heart attack.")
else:
    print("The patient is not likely to have a heart attack.")
```
 -  The result was ```The patient is not likely to have a heart attack.``` for male aged 20.

When we resampled the data and made predictions, we changed ```Gender': [0]``` and our result was ```The patient is likely to have a heart attack.``` for female aged 20.

When changed our data parameters to
```
    'Age': [60],
    'Gender': [1]
```
Our result was  ```The patient is likely to have a heart attack.``` for male aged 60.

## Conclusion
This project demonstrates a complete workflow for heart attack prediction using machine learning, including data preprocessing, feature engineering, and model training.The Random Forest Classifier was used for classification, and hyperparameter tuning was performed to improve model performance.
