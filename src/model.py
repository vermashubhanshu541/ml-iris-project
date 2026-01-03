import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = pd.read_csv('data/Iris.csv')
print(data.head())

#Features (input)
x = data.drop(columns=['Species', 'Id'])

#Target (output)
y = data['Species']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=200)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')

feature_cols = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
new_flowers_df = pd.DataFrame([[5.1, 3.5, 1.4, 0.2]], columns=feature_cols)
predictions = model.predict(new_flowers_df)
print("Predictions for new flowers:", predictions)

# git init
# git add .
# git commit -m "Initial commit of Iris classification model"
# git remote add origin <your-repo-url>
# git push -u origin master