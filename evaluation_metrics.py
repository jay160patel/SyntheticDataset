import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


def preprocess_features(df):
    y = None
    for col in df.columns:
        if col.lower() in ["target", "label", "loanapproved", "movement"]:
            y = df[col]
            X = df.drop(columns=[col])
            break
    else:
        print("⚠️ No label column found!")
        return None, None, None

    if y.dtype == 'object' or y.dtype.name == 'category':
        le = LabelEncoder()
        y = le.fit_transform(y)

    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

    return X, y, preprocessor


def run_model(model, X, y, preprocessor):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = Pipeline([
        ('preprocess', preprocessor),
        ('classifier', model)
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average='macro', zero_division=0),
        "Recall": recall_score(y_test, y_pred, average='macro', zero_division=0),
        "F1 Score": f1_score(y_test, y_pred, average='macro', zero_division=0)
    }


def evaluate_models(real_df, synthetic_df, domain):
    def evaluate(df, label):
        X, y, preprocessor = preprocess_features(df)
        if X is None or y is None or preprocessor is None:
            print(f"❌ Failed to prepare data for {label}. Skipping...")
            return None

        return {
            'Decision Tree': run_model(DecisionTreeClassifier(), X, y, preprocessor),
            'Random Forest': run_model(RandomForestClassifier(), X, y, preprocessor),
            'Logistic Regression': run_model(LogisticRegression(max_iter=1000), X, y, preprocessor),
            'XGBoost': run_model(XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'), X, y, preprocessor)
        }

    print("📌 Decision Tree / Random Forest / Logistic Regression / XGBoost Results:\n")

    real_results = evaluate(real_df, "Real")
    synth_results = evaluate(synthetic_df, "Synthetic")

    if real_results:
        print("🔷 Real Data:")
        for model, scores in real_results.items():
            print(f"  {model}:")
            for k, v in scores.items():
                print(f"    {k}: {v:.2f}")

    if synth_results:
        print("\n🟡 Synthetic Data:")
        for model, scores in synth_results.items():
            print(f"  {model}:")
            for k, v in scores.items():
                print(f"    {k}: {v:.2f}")
