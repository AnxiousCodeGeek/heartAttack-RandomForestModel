# Heart Attack Prediction: Random Forest Model

## Introduction
This repository contains a project for predicting heart attacks using a dataset of patient information. The project includes data preprocessing, feature engineering, model training using **Random Forest Classifier**.

## Requirements
 - ```pandas```
 - ```scikit-learn```
 - ```matplotlib```
 - ```seaborn```
 - ```tensorflow```
 - ```keras```
 - ```numpy```

## Working
 - Split features and target variables and using Label Encoder for categorical data:
```python
X = cleaned_df[['Age', 'Gender', 'Heart rate', 'Systolic blood pressure', 'Diastolic blood pressure', 'weight', 'height', 'BMI', 'pulse_pressure', 'BP_based_Condition']]
y = cleaned_df['Result']
le = LabelEncoder()
X['BP_based_Condition'] = le.fit_transform(X['BP_based_Condition'])
```
 - Splitting data into test and train
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

 - Training a Random Forest Classifier
```python
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
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
## Conclusion
This project demonstrates a complete workflow for heart attack prediction using machine learning, including data preprocessing, feature engineering, and model training.The Random Forest Classifier was used for classification, and hyperparameter tuning was performed to improve model performance.
