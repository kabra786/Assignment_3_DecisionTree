print("Program Success")

import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Load dataset
input_file = 'data_decision_trees.txt'

data = np.loadtxt(input_file, delimiter=',', dtype=str)

# Separate input and output
X = data[:, :-1].astype(float)
y = data[:, -1].astype(int)

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=5
)

# Create Decision Tree Classifier
params = {
    'random_state': 0,
    'max_depth': 4
}

classifier = DecisionTreeClassifier(**params)

# Train model
classifier.fit(X_train, y_train)

# Predict test data
y_test_pred = classifier.predict(X_test)

# Class names
class_names = [
    'Low Efficiency',
    'Moderate',
    'High Efficiency'
]

# Training performance
print("\n" + "#" * 40)
print("Classifier performance on Training Dataset")
print("#" * 40)

print(
    classification_report(
        y_train,
        classifier.predict(X_train),
        target_names=class_names,
        zero_division=0
    )
)

# Testing performance
print("\n" + "#" * 40)
print("Classifier performance on Test Dataset")
print("#" * 40)

print(
    classification_report(
        y_test,
        y_test_pred,
        target_names=class_names,
        zero_division=0
    )
)

print("#" * 40)
print("Decision Tree Classification Completed Successfully!")