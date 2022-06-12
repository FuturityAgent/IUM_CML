import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    feature_names = ["BMI", "SleepTime", "Sex", "Diabetic", "PhysicalActivity", "Smoking", "AlcoholDrinking", "HeartDisease"]

    dataset = pd.read_csv('heart_2020_cleaned.csv')
    dataset = dataset.dropna()

    dataset["Diabetic"] = dataset["Diabetic"].apply(lambda x: int("Yes" in x))
    dataset["HeartDisease"] = dataset["HeartDisease"].apply(lambda x: int(x == "Yes"))
    dataset["PhysicalActivity"] = dataset["PhysicalActivity"].apply(lambda x:  int(x == "Yes"))
    dataset["Smoking"] = dataset["Smoking"].apply(lambda x: (x == "Yes"))
    dataset["AlcoholDrinking"] = dataset["AlcoholDrinking"].apply(lambda x: int(x == "Yes"))
    dataset["Sex"] = dataset["Sex"].apply(lambda x: 1 if x == "Female" else 0)

    dataset = dataset[feature_names]
    dataset_train, dataset_test = train_test_split(dataset, test_size=.1, train_size=.9, random_state=1)

    dataset_train.to_csv("training_data.csv")
    dataset_test.to_csv("test_data.csv")


main()
