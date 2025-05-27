# Task_2-Elevate-Labs-
 Exploratory Data Analysis (EDA) [Data visualization, descriptive statistics, pattern recognition.]
 
1. Import Libraries
It imports various libraries for data manipulation, visualization, preprocessing, machine learning, and model saving, such as:
pandas for data manipulation.
numpy for numerical operations.
matplotlib and seaborn for data visualization.
sklearn utilities and modules for preprocessing, model selection, cross-validation, hyperparameter tuning, classification, and evaluation.
joblib for saving the trained model.

2. Read Data and Initial Exploration
Reads a CSV file into a Pandas DataFrame (df) and prints basic information:
Data types of each column.
Number of missing values in each column.
Descriptive statistics of numeric columns.
Column means for numeric features.

3. Handling Missing Data
For Age: Imputes missing values using the median.
For Embarked: Imputes missing values using the most frequent category.
Drops the Cabin column, likely due to a high percentage of missing values.

4. Encoding Categorical Variables
Converts non-numeric categorical columns (Sex, Embarked) into numeric representations using LabelEncoder.

5. Scaling Numeric Features
Scales numerical columns (Age, Fare, SibSp, Parch) using StandardScaler to normalize their distributions.

6. Visualizations
Generates boxplots for numeric features to detect outliers.
Plots histograms to observe the distribution of numeric features.
Removes outliers for numeric columns using the Interquartile Range (IQR) method.
Plots a correlation heatmap for numerical features.
Displays the target class (Survived) distribution using a count plot.

7. Final Data Preprocessing
Drops certain columns that aren't useful for modeling (PassengerId, Name, Ticket, and the already dropped Cabin).
Drops leftover non-numeric or redundant columns, printing the remaining columns and their data types.

8. Splitting Data for Modeling
Defines features (X) as all columns except the target (Survived) and defines the target (y) as the Survived column.
Splits the dataset into training and testing sets using an 80-20 train-test split.

9. Model Training and Cross-Validation
Initializes a RandomForestClassifier with a fixed random seed for reproducibility.
Performs 5-fold cross-validation on the training set and prints mean CV accuracy.

10. Hyperparameter Tuning using GridSearchCV
Defines a hyperparameter grid (param_grid) for a RandomForestClassifier including:
n_estimators: Number of trees in the forest.
max_depth: Maximum depth of each tree.
min_samples_split: Minimum samples to split a node.
Performs a grid search with 3-fold cross-validation to find the best hyperparameters.
Outputs the best parameters and best CV score.

11. Evaluate the Model
Retrains a RandomForestClassifier using the best parameters on the training data.
Evaluates its performance on the test set:
Accuracy score.
Classification report (precision, recall, F1-score for each class).
Confusion matrix, visualized as a heatmap.
Plots the feature importances of the model.

12. Save the Model
Saves the trained RandomForestClassifier (with tuned hyperparameters) to a .pkl file (titanic_rf_model.pkl) using joblib.

13. Final Dataset Info
Prints the final cleaned dataset's shape and the first few rows for inspection.

It is well-structured and likely aimed at predicting survival (Survived) based on passenger attributes in the Titanic dataset or a similar dataset.
