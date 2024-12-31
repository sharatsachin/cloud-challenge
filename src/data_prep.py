import argparse
import json
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import mlflow
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def preprocess_data(data):
    # Feature engineering similar to the original script
    def get_title(name):
        if "." in name:
            return name.split(",")[1].split(".")[0].strip()
        else:
            return "Unknown"

    def set_title(x):
        title = x["Title"]
        if title in ["Capt", "Col", "Don", "Jonkheer", "Major", "Rev", "Sir"]:
            return "Mr"
        elif title in ["the Countess", "Mme", "Lady", "Dona"]:
            return "Mrs"
        elif title in ["Mlle", "Ms"]:
            return "Miss"
        elif title == "Dr":
            return "Mr" if x["Sex"] == "male" else "Mrs"
        else:
            return title

    # Preprocessing steps
    data["Last_Name"] = data["Name"].apply(lambda x: str.split(x, ",")[0])
    data["Fare"] = data["Fare"].fillna(data["Fare"].mean())
    data["Embarked"] = data["Embarked"].fillna(data["Embarked"].mode()[0])

    # Title processing
    data["Title"] = data["Name"].map(get_title)
    data["Title"] = data.apply(set_title, axis=1)

    # Age processing
    data["Age"] = data.groupby("Title")["Age"].transform(lambda x: x.fillna(x.median()))

    # Categorical encoding
    lbl = LabelEncoder()
    data["Age"] = pd.qcut(data["Age"], 4)
    data["Age"] = lbl.fit_transform(data["Age"])

    data["Fare"] = pd.qcut(data["Fare"], 4)
    data["Fare"] = lbl.fit_transform(data["Fare"])

    data["Title"] = (
        data["Title"].map({"Mr": 0, "Miss": 1, "Mrs": 2, "Master": 3}).fillna(-1)
    )
    data["Sex"] = data["Sex"].map({"male": 0, "female": 1}).fillna(-1)
    data["Embarked"] = data["Embarked"].map({"S": 0, "C": 1, "Q": 2}).fillna(-1)

    # Cabin processing
    data["Cabin"] = data["Cabin"].fillna("Unknown")
    data["Cabin"] = data["Cabin"].map(lambda x: x[0])
    data["Cabin"] = data["Cabin"].apply(lambda x: 0 if x == "U" else 1)

    # Family size feature
    data["Family_Size"] = data["SibSp"] + data["Parch"]

    # Drop unnecessary columns
    data = data.drop(["Name", "Parch", "SibSp", "Ticket", "Last_Name"], axis=1)

    return data


def save_processed_data(train_processed, test_processed, data_path):
    # Save processed data
    if not os.path.exists(os.path.join(data_path, "feature")):
        logger.info("Creating feature directory as it does not exist")
        os.makedirs(os.path.join(data_path, "feature"))

    train_processed.to_parquet(
        os.path.join(data_path, "feature/train.parquet"), index=False
    )
    test_processed.to_parquet(
        os.path.join(data_path, "feature/test.parquet"), index=False
    )


def prepare_data(data_path):
    mlflow.set_experiment("Titanic_Data_Preparation")
    logging.info("Entering data preparation script")

    with mlflow.start_run(run_name="data_prep"):
        logger.info("Loading data")
        train = pd.read_csv(os.path.join(data_path, "raw/train.csv"))
        test = pd.read_csv(os.path.join(data_path, "raw/test.csv"))

        logger.info("Getting the ID column")
        ntrain = train.shape[0]
        passId = test["PassengerId"]
        data = pd.concat((train, test))

        logger.info("Preprocessing data")
        processed_data = preprocess_data(data)

        logger.info("Splitting data")
        train_processed = processed_data[:ntrain]
        test_processed = processed_data[ntrain:]

        logger.info("Logging data shapes to MLflow")
        mlflow.log_param("train_data_shape", train_processed.shape)
        mlflow.log_param("test_data_shape", test_processed.shape)

        logger.info("Saving processed data")
        save_processed_data(train_processed, test_processed, data_path)
        return train_processed, test_processed, passId


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_args", default='{"data_path": "data/"})')
    args = parser.parse_args()
    args = json.loads(args.json_args)
    logger.info(f"args: {args}")
    prepare_data(args["data_path"])


if __name__ == "__main__":
    main()
