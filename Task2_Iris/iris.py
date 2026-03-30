# import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# import sklearn modules
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# load dataset
df = pd.read_csv("iris.csv")

# display first rows
print("First 5 rows:\n", df.head())

# dataset information
print("\nDataset shape:", df.shape)
print("\nColumns:", df.columns)

# check missing values
print("\nMissing values:\n", df.isnull().sum())

# visualize class distribution
plt.figure(figsize=(6,4))
sns.countplot(x=df.iloc[:, -1])
plt.title("Class Distribution")
plt.show()

# pairplot for visualization
sns.pairplot(df, hue=df.columns[-1])
plt.show()

# separate features and target
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# encode target labels (string to number)
le = LabelEncoder()
y = le.fit_transform(y)

# feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# build model
model = DecisionTreeClassifier(max_depth=4)

# train model
model.fit(X_train, y_train)

# make predictions
y_pred = model.predict(X_test)

# evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy:", accuracy)

# classification report
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)

# plot confusion matrix
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# feature importance
importances = model.feature_importances_
features = X.columns

plt.figure(figsize=(6,4))
plt.barh(features, importances)
plt.title("Feature Importance")
plt.xlabel("Importance Score")
plt.show()

# sample prediction
sample = np.array([[5.1, 3.5, 1.4, 0.2]])

sample_scaled = scaler.transform(sample)

prediction = model.predict(sample_scaled)

# decode prediction back to label
predicted_label = le.inverse_transform(prediction)

print("\nSample Prediction:", predicted_label[0])
