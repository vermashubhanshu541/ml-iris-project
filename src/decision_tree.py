import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
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

# Train Decision Tree Classifier
tree = DecisionTreeClassifier(random_state=42, max_depth=3)  # Limit depth to avoid overfitting
tree.fit(X_train, y_train)
y_pred_tree = tree.predict(X_test)

print(f"Decision Tree Accuracy: {accuracy_score(y_test, y_pred_tree):.3f}")
print(classification_report(y_test, y_pred_tree))

# Feature importance (how much each feature helped splits)
print("Feature Importances:", dict(zip(['SL','SW','PL','PW'], tree.feature_importances_)))