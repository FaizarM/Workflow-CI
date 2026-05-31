import argparse
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

TARGET = "NObeyesdad"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="obesity_preprocessing.csv")
    parser.add_argument("--n_estimators", type=int, default=300)
    parser.add_argument("--max_depth", type=int, default=0)
    args = parser.parse_args()

    df = pd.read_csv(args.data_path)
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    max_depth = None if args.max_depth <= 0 else args.max_depth

    mlflow.sklearn.autolog()
    with mlflow.start_run() as run:
        model = RandomForestClassifier(
            n_estimators=args.n_estimators, max_depth=max_depth, random_state=42
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        mlflow.log_metric("test_accuracy", accuracy)
        mlflow.log_metric("test_f1", f1)

        with open("run_id.txt", "w") as f:
            f.write(run.info.run_id)

        print(f"Accuracy: {accuracy:.4f} | F1: {f1:.4f}")
        print(f"Run ID: {run.info.run_id}")


if __name__ == "__main__":
    main()
