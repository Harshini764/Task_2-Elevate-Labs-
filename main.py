import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import joblib

df = pd.read_csv(r"C:\Users\Admin\Desktop\Dataset.csv", encoding='utf-8')
print("Initial Info:")
print(df.info())
print("\nMissing values:\n", df.isnull().sum())
print("\nData Types:\n", df.dtypes)
print("\nSummary statistics:\n", df.describe())
print("\nMeans:\n", df.mean(numeric_only=True))
age_imputer = SimpleImputer(strategy='median')
df['Age'] = age_imputer.fit_transform(df[['Age']]).ravel()
embarked_imputer = SimpleImputer(strategy='most_frequent')
df['Embarked'] = embarked_imputer.fit_transform(df[['Embarked']]).ravel()
df.drop(columns=['Cabin'], inplace=True)
label_encoders = {}
categorical_cols = ['Sex', 'Embarked']
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le
scaler = StandardScaler()
numerical_cols = ['Age', 'Fare', 'SibSp', 'Parch']
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
plt.figure(figsize=(12, 8))
for i, col in enumerate(numerical_cols, 1):
    plt.subplot(2, 2, i)
    sns.boxplot(x=df[col])
    plt.title(f'Boxplot of {col}')
plt.tight_layout()
plt.show()
df[numerical_cols].hist(figsize=(10, 8), bins=20)
plt.suptitle('Histograms of Numeric Features')
plt.tight_layout()
plt.show()
for col in numerical_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
numeric_df = df.select_dtypes(include=[np.number])
plt.figure(figsize=(10, 8))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()
print("Survival value counts:\n", df['Survived'].value_counts())
sns.countplot(x='Survived', data=df)
plt.title('Survival Distribution')
plt.show()
columns_to_drop = [col for col in ['PassengerId', 'Name', 'Ticket', 'Cabin'] if col in df.columns]
df.drop(columns=columns_to_drop, inplace=True)
for col in df.columns:
    if df[col].dtype == 'object':
        print(f"Dropping non-numeric column: {col}")
        df.drop(columns=[col], inplace=True)
print("Columns before modeling:", df.columns.tolist())
print("Data types before modeling:\n", df.dtypes)
print("Sample data:\n", df.head())
bad_cols = [col for col in df.columns if col != 'Survived' and not np.issubdtype(df[col].dtype, np.number)]
if bad_cols:
    print("Dropping non-numeric columns (final check):", bad_cols)
    df.drop(columns=bad_cols, inplace=True)
X = df.drop('Survived', axis=1)
y = df['Survived']
print("Columns used for modeling:", X.columns.tolist())
print("Data types used for modeling:\n", X.dtypes)
print("Sample X data:\n", X.head())
X = df.drop('Survived', axis=1)
y = df['Survived']
print("Columns used for modeling:", X.columns.tolist())
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Train shape:", X_train.shape, "Test shape:", X_test.shape)
clf = RandomForestClassifier(random_state=42)
cv_scores = cross_val_score(clf, X, y, cv=5)
print("Cross-validation scores:", cv_scores)
print("Mean CV accuracy:", np.mean(cv_scores))
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10]
}
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3)
grid_search.fit(X_train, y_train)
print("Best parameters:", grid_search.best_params_)
print("Best CV score:", grid_search.best_score_)
best_clf = grid_search.best_estimator_
best_clf.fit(X_train, y_train)
y_pred = best_clf.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.show()
importances = best_clf.feature_importances_
feat_names = X.columns
plt.figure(figsize=(8, 5))
sns.barplot(x=importances, y=feat_names)
plt.title('Feature Importances')
plt.show()
joblib.dump(best_clf, 'titanic_rf_model.pkl')
print("\nCleaned dataset shape:", df.shape)
print(df.head())
