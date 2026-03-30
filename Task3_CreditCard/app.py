# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# sklearn modules
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# load dataset
df = pd.read_csv("creditcard.csv")

# display first rows
print("First 5 rows:\n", df.head())

# dataset info
print("\nDataset shape:", df.shape)
print("\nMissing values:\n", df.isnull().sum())

# check class distribution (important for fraud detection)
print("\nClass Distribution:\n", df['Class'].value_counts())

# visualize class imbalance
plt.figure(figsize=(6,4))
sns.countplot(x='Class', data=df)
plt.title("Fraud vs Normal Transactions")
plt.show()

# separate features and target
X = df.drop('Class', axis=1)
y = df['Class']

# scale features (important)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# build model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# train model
model.fit(X_train, y_train)

# predictions
y_pred = model.predict(X_test)

# accuracy
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy)

# classification report
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)

# visualize confusion matrix
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# feature importance
importances = model.feature_importances_
features = X.columns

plt.figure(figsize=(8,6))
plt.barh(features[:10], importances[:10])
plt.title("Top Feature Importance")
plt.xlabel("Importance Score")
plt.show()

# sample prediction (random transaction)
sample = X.iloc[0:1]
sample_scaled = scaler.transform(sample)

prediction = model.predict(sample_scaled)

print("\nSample Prediction (0=Normal, 1=Fraud):", prediction[0])
