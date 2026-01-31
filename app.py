from src.data_preprocessing import load_data, preprocess_data
from src.model_training import train_model
from src.prediction import predict_proba
from sklearn.metrics import accuracy_score, roc_auc_score

df = load_data("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
df = preprocess_data(df)

model, X_test, y_test = train_model(df)

y_probs = predict_proba(model, X_test)
y_pred = (y_probs >= 0.5).astype(int)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_probs))
