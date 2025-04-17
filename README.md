Cosmic Classifier
=================

Overview
--------

Our project uses the **CatBoostClassifier**, a gradient boosting algorithm that handles categorical features natively. The dataset includes features such as magnetic field levels, radiation levels, and other planetary attributes, which are preprocessed and used to train the model. The final model predicts the category of planets in the test dataset.

Installation
------------

To run this project, you need to install the following Python libraries:



`   !pip install catboost scikit-learn optuna   `

Code Explanation
----------------

### 1. ***Loading the Dataset***

*   The dataset is loaded from CSV files (cosmicclassifierTraining.csv and cosmicclassifierTest.csv) using pandas.
    


`   from google.colab import files  uploaded = files.upload()  train_data = pd.read_csv("cosmicclassifierTraining.csv")  uploaded = files.upload()  test_data = pd.read_csv("cosmicclassifierTest.csv")   `

### 2. ***Preprocessing the Data***

*   **Column Name Cleaning:** Spaces in column names are removed to avoid issues during data manipulation.
    
*   **Handling Missing/Noisy Data:** Rows with large negative values (-999999) or NaN values are removed to clean the dataset.
    
*   **Separate Features and Target:** The target column (Prediction) is separated from the features.
    
*   **Identify Categorical Columns:** Columns containing the word "Category" or with an object data type are identified as categorical.
    


`   train_data.columns = train_data.columns.str.replace(" ", "")  test_data.columns = test_data.columns.str.replace(" ", "")  train_data = train_data[(train_data != -999999).all(axis=1)]  train_data = train_data.dropna()  test_data = test_data[(test_data != -999999).all(axis=1)]  test_data = test_data.dropna()  X = train_data.drop(columns=["Prediction"], errors='ignore')  y = train_data["Prediction"]  cat_cols = [col for col in X.columns if "Category" in col or X[col].dtype == "object"]  cat_indices = [X.columns.get_loc(col) for col in cat_cols]   `

### 3. ***Normalization***

*   Numeric features are normalized using MinMaxScaler to scale values between 0 and 1.
    
*   Any remaining NaN values in numeric columns are filled with 0.
    


`   scaler = MinMaxScaler()  num_cols = [col for col in X.columns if col not in cat_cols]  X[num_cols] = scaler.fit_transform(X[num_cols])  test_data[num_cols] = scaler.transform(test_data[num_cols])  for col in num_cols:      X[col] = pd.to_numeric(X[col], errors="coerce")      test_data[col] = pd.to_numeric(test_data[col], errors="coerce")  X[num_cols] = X[num_cols].fillna(0)  test_data[num_cols] = test_data[num_cols].fillna(0)   `

### 4. ***Data Splitting***

*   The dataset is split into training and validation sets using an 80-20 split, with stratification to maintain the distribution of the target variable.

    
`   X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y)   `

### 5. ***Hyperparameter Tuning with Optuna***

*   **Objective Function:** An objective function is defined to optimize the hyperparameters of the CatBoostClassifier. The function suggests values for key hyperparameters such as iterations, learning\_rate, depth, l2\_leaf\_reg, and others.
    
*   **Optuna Study:** An Optuna study is created with the **TPE (Tree-structured Parzen Estimator) sampler** for efficient hyperparameter search and **HyperbandPruner** for early stopping of unpromising trials. The study runs for 50 trials to find the best hyperparameters.
    

`   def objective(trial):      params = {          'iterations': trial.suggest_int('iterations', 100, 1000),          'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),          'depth': trial.suggest_int('depth', 3, 11),          'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 2, 10),          'border_count': trial.suggest_int('border_count', 32, 255),          'random_strength': trial.suggest_float('random_strength', 0.1, 10),          'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),          'random_state': 42,          'verbose': False,      }      model = CatBoostClassifier(**params)      model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50, verbose=False)      y_val_pred = model.predict(X_val)      accuracy = accuracy_score(y_val, y_val_pred)      return accuracy  study = optuna.create_study(      direction='maximize',      sampler=TPESampler(),      pruner=HyperbandPruner()  )  study.optimize(objective, n_trials=50)  print("Best hyperparameters:", study.best_params)   `

### 6. ***Model Training***

*   A **CatBoostClassifier** is initialized with the best hyperparameters found by Optuna.
    
*   The model is trained on the training set and evaluated on the validation set.
    

`   best_params = study.best_params  final_model = CatBoostClassifier(**best_params, random_state=42, verbose=False)  final_model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50, verbose=False)   `

### 7. ***Model Evaluation***

*   The model's performance is evaluated on the validation set using accuracy as the metric.
    

`   y_val_pred = final_model.predict(X_val)  accuracy = accuracy_score(y_val, y_val_pred)  print(f"Validation Accuracy with Best Hyperparameters: {accuracy * 100:.2f}%")   `

### 8. ***Final Model Training***

*   The model is retrained on the full dataset (training + validation) to improve generalization.
    

`   final_model.fit(X, y)   `

### 9. ***Prediction and Submission***

*   The trained model predicts the categories of planets in the test dataset.
    
*   Predictions are mapped to human-readable planet categories and saved in a CSV file for submission.
    

`   test_predictions = final_model.predict(test_data)  planet_categories = {      0.0: "Bewohnbar",      1.0: "Terraformierbar",      2.0: "Rohstoffreich",      3.0: "Wissenschaftlich",      4.0: "Gasriese",      5.0: "Wüstenplanet",      6.0: "Eiswelt",      7.0: "Toxischetmosäre",      8.0: "Hohestrahlung",      9.0: "Toterahswelt"  }  test_predictions_categories = [planet_categories[pred] for pred in test_predictions]  submission = pd.DataFrame({      "Planet_ID": test_data.index,      "Predicted_Class": test_predictions_categories  })  submission.to_csv("submission.csv", index=False)  files.download("submission.csv")   `

Results
-------

The final model's predictions are saved in submission.csv, which contains the predicted categories for each planet in the test dataset. The validation accuracy achieved during training is also printed.

License
-------

This project is open-source.
