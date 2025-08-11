# -----------------------
# Telecom Churn Analysis
# -----------------------

# 1. Import Libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, RocCurveDisplay

# 2. Load Dataset
df = pd.read_csv("Telco-Customer-Churn.csv")
print("First 5 rows:")
print(df.head())
print("\nDataset Info:")
print(df.info())

# 3. Encode Categorical Data
encoder = LabelEncoder()
for col in df.select_dtypes(include=['object']).columns:
    df[col] = encoder.fit_transform(df[col])

# 4. Split Data
X = df.drop('Churn', axis=1)
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train Model (Random Forest)
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

# 6. Evaluate
y_pred = rf.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 7. ROC Curve
RocCurveDisplay.from_estimator(rf, X_test, y_test)
plt.title("ROC Curve - Random Forest")
plt.show()

# 8. Feature Importance
importance = pd.Series(rf.feature_importances_, index=X.columns)
importance.sort_values(ascending=False).head(10).plot(kind='bar', color='skyblue')
plt.title("Top 10 Feature Importances")
plt.show()
