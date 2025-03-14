import sys

sys.path.append("..")

from datasets import datasets

import argparse

import numpy as np

import os
import subprocess

from aeon.datasets import load_from_ts_file
from aeon.benchmarking.resampling import stratified_resample_data
from aeon.classification.distance_based import KNeighborsTimeSeriesClassifier
from sklearn.metrics import accuracy_score


def benchmark(name, link, seed=0):

    # Download dataset
    dataset_path = os.path.join("datasets", name)

    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path, exist_ok=True)

        zip_path = os.path.join(dataset_path, "data.zip")
        subprocess.run(["wget", "-nv", "-O", zip_path, link], check=True)
        subprocess.run(["unzip", "-q", "-o", zip_path, "-d", dataset_path], check=True)
        os.remove(zip_path)

    print(f"Downloaded {name}")

    # Load dataset
    train_file = os.path.join(dataset_path, f"{name}_TRAIN.ts")
    X_train, y_train = load_from_ts_file(train_file, return_type="numpy3D")

    test_file = os.path.join(dataset_path, f"{name}_TEST.ts")
    X_test, y_test = load_from_ts_file(test_file, return_type="numpy3D")

    y_train = np.unique(y_train, return_inverse=True)[1] + 1
    y_test = np.unique(y_test, return_inverse=True)[1] + 1

    all_acc = []

    for e in range(1, 31):
        print(f"{name} - Epoch {e}")

        if e == 1:
            X_train_c = X_train.copy()
            X_test_c = X_test.copy()
            y_train_c = y_train.copy()
            y_test_c = y_test.copy()
        else:
            X_train_c, y_train_c, X_test_c, y_test_c = stratified_resample_data(
                X_train, y_train, X_test, y_test, random_state=e
            )

        trainer = KNeighborsTimeSeriesClassifier(n_jobs=-1)

        trainer.fit(X_train_c, y_train_c)

        y_pred = trainer.predict(X_test_c)

        all_acc.append(accuracy_score(y_pred, y_test_c))

    all_acc = np.array(all_acc)
    print(f"Accuracy: {np.mean(all_acc)}")
    print(f"Std: {np.std(all_acc)}")

    return np.mean(all_acc), np.std(all_acc)


parser = argparse.ArgumentParser(description="Process dataset ID.")
parser.add_argument("--id", type=int, required=True, help="ID of the dataset")
args = parser.parse_args()

dataset_list = list(datasets.keys())

seed = 0
np.random.seed(seed)

os.makedirs("datasets", exist_ok=True)

results_file = "results_knndtw.txt"

try:
    dataset_index = args.id
    if dataset_index < 1 or dataset_index > len(dataset_list) + 1:
        raise ValueError("Dataset number out of range")

    dataset_name = dataset_list[dataset_index - 1]  # Convert number to dataset name
    dataset_link = datasets[dataset_name]

    print(f"Starting benchmark for dataset {dataset_name}")

    mean, std = benchmark(dataset_name, dataset_link, seed=seed, device="cuda")

    with open(results_file, "a") as f:
        f.write(f"{dataset_name}: mean={mean:.4f}, std={std:.4f}\n")

except ValueError as e:
    print(f"Error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
