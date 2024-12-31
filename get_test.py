import pandas as pd

import re

test_data_with_labels = pd.read_csv("./data/raw/titanic.csv")
test_data = pd.read_csv("./data/raw/test.csv")

test_data

test_data_with_labels

for i, name in enumerate(test_data_with_labels["name"]):
    if '"' in name:
        test_data_with_labels["name"][i] = re.sub('"', "", name)

for i, name in enumerate(test_data["Name"]):
    if '"' in name:
        test_data["Name"][i] = re.sub('"', "", name)

test_data["Name_dup"] = test_data["Name"]


for i, name in enumerate(test_data_with_labels["name"]):
    if '"' in name:
        test_data_with_labels["name"][i] = re.sub('"', "", name)

for i, name in enumerate(test_data["Name_dup"]):
    if '"' in name:
        test_data["Name_dup"][i] = re.sub('"', "", name)

survived = []

for name in test_data["Name"]:
    survived.append(
        int(
            test_data_with_labels.loc[test_data_with_labels["name"] == name][
                "survived"
            ].values[-1]
        )
    )

test_data["Survived"] = survived
test_data = test_data.drop(columns=["Name_dup"])


test_data.to_csv("./data/raw/test_l.csv", index=False)
