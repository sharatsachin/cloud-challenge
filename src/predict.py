import argparse
import json
import os
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn

import logging
import warnings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calculate_test_score(df):
    # Calculate the test score
    test_score = np.mean(df["Survived_pred"] == df["Survived_actual"])
    return test_score


def save_processed_data(test_processed, data_path):
    # Save processed data
    if not os.path.exists(os.path.join(data_path, "output")):
        logger.info("Creating output directory as it does not exist")
        os.makedirs(os.path.join(data_path, "output"))

    test_processed.to_parquet(
        os.path.join(data_path, "output/test_pred.parquet"), index=False
    )
    logger.info("Predictions on test data saved to test_pred.parquet")
    return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_args", default='{"data_path": "data/"})')
    args = parser.parse_args()
    args = json.loads(args.json_args)
    logger.info(f"args: {args}")

    runs = mlflow.search_runs(
        experiment_names=["Titanic_Model_Training"],
        filter_string="tags.mlflow.runName = 'train_models'",
        order_by=["end_time DESC"],
    )
    run_id = runs.iloc[0].run_id
    scaler_uri = f"runs:/{run_id}/scaler"
    model_uri = f"runs:/{run_id}/Soft_Voting_Classifier"

    test_processed = pd.read_parquet(
        os.path.join(args["data_path"], "feature/test.parquet")
    )
    passId = test_processed["PassengerId"]

    y_test = test_processed["Survived"].values
    X_test = test_processed.drop(columns=["PassengerId", "Survived"])

    scaler = mlflow.sklearn.load_model(scaler_uri)

    X_test = scaler.transform(test_processed)

    # Load the best model (soft voting classifier)
    model = mlflow.sklearn.load_model(model_uri)

    # Make predictions
    with warnings.catch_warnings(action="ignore"):
        submission = model.predict(X_test)

    test_processed["Survived_pred"] = submission
    test_processed["Survived_actual"] = y_test
    test_processed["PassengerId"] = passId

    test_score = calculate_test_score(test_processed)
    logger.info(f"Test score: {test_score:.4f}")

    with mlflow.start_run(run_id=run_id):
        mlflow.log_metric("test_score", test_score)

    save_processed_data(test_processed, args["data_path"])


if __name__ == "__main__":
    main()
