# ----- Classification: Pass/Fail prediction -----

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1. Sample dataset (features: [hours_studied, practice_questions], labels: 0=Fail, 1=Pass)
X = [
    [1, 5],   # low study, low practice
    [2, 10],
    [3, 15],
    [4, 20],
    [5, 25],
    [6, 30],
    [7, 35],
    [8, 40],
    [9, 45],
    [10, 50]
]

y = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]  # corresponding labels

# 2. Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 3. Create model (Logistic Regression for classification)
model = LogisticRegression()


# 4. Train the model
model.fit(X_train, y_train)

# 5. Predict on test set
y_pred = model.predict(X_test)


# 6. Evaluate accuracy
acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)

# 7. Predict for a new student
new_student = [[4, 18]]  # 4 hours study, 18 practice questions
prediction = model.predict(new_student)

if prediction[0] == 1:
    print("Model prediction: PASS")
else:
    print("Model prediction: FAIL")


'''What’s happening:

LogisticRegression() is a classification algorithm (despite the name “regression”).

.fit(X_train, y_train) trains the model.

.predict(X_test) gives predicted labels.

accuracy_score tells how many predictions were correct.'''