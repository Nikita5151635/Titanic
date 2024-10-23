import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib  # Новый импорт

# 1. Загрузка данных
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# 2. Предобработка данных
train_data['Age'] = train_data['Age'].fillna(train_data['Age'].median())
test_data['Age'] = test_data['Age'].fillna(test_data['Age'].median())
test_data['Fare'] = test_data['Fare'].fillna(test_data['Fare'].median())

train_data = train_data.drop('Cabin', axis=1)
test_data = test_data.drop('Cabin', axis=1)
train_data = train_data.drop('Ticket', axis=1)
test_data = test_data.drop('Ticket', axis=1)

train_data['Embarked'] = train_data['Embarked'].fillna(train_data['Embarked'].mode()[0])

label = LabelEncoder()
train_data['Sex'] = label.fit_transform(train_data['Sex'])
test_data['Sex'] = label.transform(test_data['Sex'])
train_data['Embarked'] = label.fit_transform(train_data['Embarked'])
test_data['Embarked'] = label.transform(test_data['Embarked'])

X = train_data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
y = train_data['Survived']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print(f'Точность модели на валидационном наборе: {accuracy * 100:.2f}%')

X_test = test_data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
test_predictions = model.predict(X_test)

submission = pd.DataFrame({
    'PassengerId': test_data['PassengerId'],
    'Survived': test_predictions
})
submission.to_csv('submission.csv', index=False)
print("Файл submission.csv успешно создан!")

# Сохранение модели
joblib.dump(model, 'titanic_model.pkl')  # Сохраняем модель
