import argparse
import json
import os
import warnings
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    ExtraTreesClassifier,
)
from mlflow.models import infer_signature
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import BaggingClassifier, VotingClassifier

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def log_mlflow_objects(X_train, final_model, model_scores, scaler):
    mlflow.set_experiment("Titanic_Model_Training")

    with warnings.catch_warnings(action="ignore"):
        signature = infer_signature(X_train, final_model.predict(X_train))

    with mlflow.start_run(run_name="train_models"):
        mlflow.log_param("Number_of_models", len(model_scores))

        for name, model, score in model_scores:
            mlflow.log_metric(f"{name}_accuracy", score)
            mlflow.sklearn.log_model(
                sk_model=model, artifact_path=name, signature=signature
            )

        model_info = mlflow.sklearn.log_model(
            sk_model=final_model,
            artifact_path="Soft_Voting_Classifier",
            signature=signature,
        )
        mlflow.sklearn.log_model(sk_model=scaler, artifact_path="scaler")

    logger.info("Model objects logged to MLflow")
    logger.info(f"Final model: {model_info.model_uri}")
    return model_info


def train_models(X_train, y_train):
    models = [
        (
            "Random Forest",
            RandomForestClassifier(
                n_estimators=25,
                max_depth=3,
                max_features=3,
                min_samples_leaf=2,
                min_samples_split=8,
                random_state=1,
            ),
        ),
        ("Logistic Regression", LogisticRegression(C=2.7825, penalty="l2")),
        (
            "LightGBM",
            LGBMClassifier(
                learning_rate=0.01, n_estimators=100, random_state=1, verbose=-1
            ),
        ),
        (
            "Gradient Boosting",
            GradientBoostingClassifier(
                learning_rate=0.0005, n_estimators=1250, random_state=1
            ),
        ),
        (
            "Extra Trees",
            ExtraTreesClassifier(
                max_depth=None,
                max_features=3,
                min_samples_leaf=2,
                min_samples_split=8,
                n_estimators=10,
                random_state=1,
            ),
        ),
        (
            "AdaBoost",
            AdaBoostClassifier(learning_rate=0.1, n_estimators=50, random_state=1),
        ),
        (
            "KNN",
            KNeighborsClassifier(
                algorithm="auto", leaf_size=1, n_neighbors=5, weights="uniform"
            ),
        ),
        ("SVC", SVC(probability=True)),
        ("Gaussian Process", GaussianProcessClassifier()),
        ("Bagging", BaggingClassifier(random_state=1)),
    ]
    model_scores = []
    for name, model in models:
        cv_scores = cross_val_score(model, X_train, y_train, cv=10, scoring="accuracy")
        mean_score = cv_scores.mean()
        model_scores.append((name, model, mean_score))

    model_scores.sort(key=lambda x: x[2], reverse=True)
    voting_models = [
        (name, model) for name, model, score in model_scores if score > 0.8
    ]
    soft_voting = VotingClassifier(estimators=voting_models, voting="soft")
    soft_voting.fit(X_train, y_train)
    return soft_voting, model_scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_args", default='{"data_path": "data/"})')
    args = parser.parse_args()
    args = json.loads(args.json_args)
    logger.info(f"args: {args}")

    logger.info("Entering model training script")

    logger.info("Loading processed data")
    train_processed = pd.read_parquet(
        os.path.join(args["data_path"], "feature/train.parquet")
    )

    logger.info("Preprocessing data")
    scaler = StandardScaler()

    y_train = train_processed["Survived"].values
    X_train = train_processed.drop(columns=["Survived", "PassengerId"])
    X_train = scaler.fit_transform(train_processed)

    logger.info("Training models")
    with warnings.catch_warnings(action="ignore"):
        final_model, model_scores = train_models(X_train, y_train)

    mlflow.set_experiment("Titanic_Model_Training")
    logger.info("Logging model objects to MLflow")
    log_mlflow_objects(X_train, final_model, model_scores, scaler)

    logger.info("Model training complete, performance metrics:")
    for name, model, score in model_scores:
        print(f"{name}: {score:.4f}")

    return final_model, scaler


if __name__ == "__main__":
    main()
