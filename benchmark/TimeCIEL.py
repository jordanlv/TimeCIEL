import sys

sys.path.append("..")

import argparse

import numpy as np
import torch

import os
import shutil
import subprocess

from aeon.datasets import load_from_ts_file
from aeon.benchmarking.resampling import stratified_resample_data
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

from torch_mas.batch.trainer import BaseTrainer
from torch_mas.batch.internal_model import NClass
from torch_mas.batch.activation_function import BaseActivation
from torch_mas.batch.trainer.learning_rules import (
    IfActivated,
    IfNoActivated,
    IfNoActivatedAndNoNeighbors,
    SimpleDestroy,
)

from torch_mas.data import DataBuffer

from datasets import datasets

import optuna
from optuna.samplers import TPESampler

optuna.logging.set_verbosity(optuna.logging.WARNING)


def create_model(
    input_dim, output_dim, seq_len, alpha, neighbor_rate, R, bad_th, n_epochs, device
):
    internal_model = NClass(
        input_dim=input_dim, output_dim=output_dim, memory_length=0, device=device
    )

    activation = BaseActivation(
        seq_len=seq_len,
        input_dim=input_dim,
        output_dim=output_dim,
        alpha=alpha,
        neighbor_rate=neighbor_rate,
        device=device,
    )

    trainer = BaseTrainer(
        activation=activation,
        internal_model=internal_model,
        R=R,
        bad_th=bad_th,
        n_epochs=n_epochs,
        learning_rules=[
            IfNoActivatedAndNoNeighbors(),
            IfNoActivated(),
            IfActivated(),
            SimpleDestroy(10000),
        ],
        device=device,
    )

    return trainer


def benchmark(name, link, n_trials=500, n_splits=3, device="cpu", seed=0):

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

    X_train = np.transpose(X_train, axes=(0, 2, 1))
    X_test = np.transpose(X_test, axes=(0, 2, 1))

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

        # Stratified KFold
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

        def objective(trial):
            _, seq_len, input_dim = X_train_c.shape

            trainer = create_model(
                input_dim,
                1,
                seq_len,
                trial.suggest_float("alpha", 0.01, 0.9),
                trial.suggest_float("neighbor_rate", 0.1, 1.0),
                [trial.suggest_float(f"R{i}", 0.01, 0.9) for i in range(input_dim)],
                trial.suggest_float("bad_th", 0.01, 0.9),
                5,
                device,
            )

            acc = []

            for train_index, test_index in skf.split(X_train_c, y_train_c):
                X_inner_train, X_inner_val = (
                    X_train_c[train_index],
                    X_train_c[test_index],
                )
                y_inner_train, y_inner_val = (
                    y_train_c[train_index],
                    y_train_c[test_index],
                )

                dataset = DataBuffer(X_inner_train, y_inner_train, device=device)

                trainer.fit(dataset)

                y_pred = []
                for batch in torch.tensor(
                    X_inner_val, dtype=torch.float32, device=device
                ).split(32):
                    y_pred += trainer.predict(batch).cpu().tolist()

                acc.append(accuracy_score(y_pred, y_inner_val))

            acc = np.array(acc)
            return np.mean(acc)

        def callback(study, trial):
            if trial.value == 1.0:
                study.stop()

        study = optuna.create_study(sampler=TPESampler(seed=seed), direction="maximize")
        study.optimize(objective, n_trials=n_trials, callbacks=[callback])

        # Evaluate
        best_params = study.best_params

        _, seq_len, input_dim = X_train_c.shape

        trainer = create_model(
            input_dim,
            1,
            seq_len,
            best_params["alpha"],
            best_params["neighbor_rate"],
            [best_params[f"R{i}"] for i in range(input_dim)],
            best_params["bad_th"],
            5,
            device,
        )

        dataset = DataBuffer(X_train_c, y_train_c, device=device)

        trainer.fit(dataset)

        y_pred = []
        for batch in torch.tensor(X_test_c, dtype=torch.float32, device=device).split(
            32
        ):
            y_pred += trainer.predict(batch).cpu().tolist()

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
torch.manual_seed(seed)

os.makedirs("datasets", exist_ok=True)

results_file = "results_timeciel.txt"

try:
    dataset_index = args.id
    if dataset_index < 1 or dataset_index > len(dataset_list) + 1:
        raise ValueError("Dataset number out of range")

    dataset_name = dataset_list[dataset_index - 1]
    dataset_link = datasets[dataset_name]

    print(f"Starting benchmark for dataset {dataset_name}")

    mean, std = benchmark(dataset_name, dataset_link, seed=seed, device="cpu")

    with open(results_file, "a") as f:
        f.write(f"{dataset_name}: mean={mean:.4f}, std={std:.4f}\n")

except ValueError as e:
    print(f"Error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
