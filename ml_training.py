import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sys


def main():
    no_of_epochs = int(sys.argv[1]) if len(sys.argv) == 2 else 10
    feature_names = ["BMI", "SleepTime", "Sex", "Diabetic", "PhysicalActivity", "Smoking", "AlcoholDrinking",
                     "HeartDisease"]

    scaler = StandardScaler()

    dataset_train = pd.read_csv("training_data.csv")
    dataset_test = pd.read_csv("test_data.csv")

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(4, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        loss=tf.keras.losses.binary_crossentropy,
        optimizer=tf.keras.optimizers.Adam(lr=0.01),
        metrics=["accuracy", tf.keras.metrics.Recall(name='recall')]
    )

    train_X = dataset_train[feature_names].astype(np.float32)
    train_Y = dataset_train["HeartDisease"].astype(np.float32)
    test_X = dataset_test[feature_names].astype(np.float32)
    test_Y = dataset_test["HeartDisease"].astype(np.float32)

    train_X = scaler.fit_transform(train_X)
    # train_Y = scaler.fit_transform(train_Y)
    test_X = scaler.fit_transform(test_X)
    # test_Y = scaler.fit_transform(test_Y)

    print(train_Y.value_counts())

    train_X = tf.convert_to_tensor(train_X)
    train_Y = tf.convert_to_tensor(train_Y)

    test_X = tf.convert_to_tensor(test_X)
    test_Y = tf.convert_to_tensor(test_Y)

    model.fit(train_X, train_Y, epochs=no_of_epochs)
    
    evaluation = model.evaluate(test_X, test_Y, return_dict=True)
    with open("train_evaluation.txt", "w") as f:
        evaluate_repr = f"ACCURACY: {evaluation.get('accuracy')}, LOSS: {evaluation.get('loss')}, RECALL: {evaluation.get('recall')}"
        
        f.write(evaluate_repr)


main()

