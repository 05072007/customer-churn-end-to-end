import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

def train_model(df):

    X = df.drop('Churn_Yes', axis=1)
    y = df['Churn_Yes']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=20,
        random_state=42
    )

    model.fit(X_train, y_train)

    joblib.dump(model, "models/model.pkl")

    return model, X_test, y_test
