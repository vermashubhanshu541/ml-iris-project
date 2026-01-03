import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# Load data
data = pd.read_csv('data/Iris.csv')
X = data.drop(['Species', 'Id'], axis=1)
y = data['Species']

# Scale features (important for logistic)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Linear kernel (straight line)
svm_linear = SVC(kernel='linear', random_state=42)
svm_linear.fit(X_train, y_train)
y_pred_linear = svm_linear.predict(X_test)
print(f"SVM Linear Accuracy: {accuracy_score(y_test, y_pred_linear):.3f}")

# RBF kernel (curvy boundary for non-linear data)
svm_rbf = SVC(kernel='rbf', random_state=42)
svm_rbf.fit(X_train, y_train)
y_pred_rbf = svm_rbf.predict(X_test)
print(f"SVM RBF Accuracy: {accuracy_score(y_test, y_pred_rbf):.3f}")