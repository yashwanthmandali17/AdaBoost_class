from preprocessing import load_and_preprocess
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import joblib
import os

os.makedirs("models", exist_ok=True)

X_train, X_test, y_train, y_test = load_and_preprocess()

# AdaBoost with Decision Tree stumps (depth=1) as base estimator
base_estimator = DecisionTreeClassifier(max_depth=1, random_state=42)

model = AdaBoostClassifier(
    estimator=base_estimator,
    n_estimators=100,
    learning_rate=0.5,
    random_state=42,
)
model.fit(X_train, y_train)

joblib.dump(model, "models/adaboost_classifier.pkl")
print("Model saved to models/adaboost_classifier.pkl")

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, y_prob):.4f}")
