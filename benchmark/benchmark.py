import sys

sys.path.append("..")

import numpy as np
import torch

import os
import shutil
import subprocess

from aeon.datasets import load_from_ts_file
from aeon.benchmarking.resampling import stratified_resample_data
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score


from torch_mas.batch.trainer import DTWTrainer
from torch_mas.batch.internal_model import NClass
from torch_mas.batch.activation_function import DTWActivation
from torch_mas.batch.trainer.learning_rules_dtw import (
    IfActivated,
    IfNoActivated,
    IfNoActivatedAndNoNeighbors,
    SimpleDestroy,
)
from torch_mas.data import DataBuffer

import optuna
from optuna.samplers import TPESampler

import tqdm

optuna.logging.set_verbosity(optuna.logging.WARNING)


def benchmark(datasets, n_trials=300, device="cpu", seed=0):

    np.random.seed(seed)
    torch.manual_seed(seed)

    os.makedirs("datasets", exist_ok=True)

    for k, v in datasets.items():
        # Download dataset
        dataset_path = os.path.join("datasets", k)

        if os.path.exists(dataset_path):
            shutil.rmtree(dataset_path)

        os.makedirs(dataset_path, exist_ok=True)

        zip_path = os.path.join(dataset_path, "data.zip")
        subprocess.run(["wget", "-nv", "-O", zip_path, v], check=True)
        subprocess.run(["unzip", "-q", "-o", zip_path, "-d", dataset_path], check=True)
        os.remove(zip_path)

        print(f"Downloaded {k}")

        # Load dataset
        train_file = os.path.join(dataset_path, f"{k}_TRAIN.ts")
        X_train, y_train = load_from_ts_file(train_file, return_type="numpy3D")

        test_file = os.path.join(dataset_path, f"{k}_TEST.ts")
        X_test, y_test = load_from_ts_file(test_file, return_type="numpy3D")

        y_train = np.unique(y_train, return_inverse=True)[1] + 1
        y_test = np.unique(y_test, return_inverse=True)[1] + 1

        X_train = np.transpose(X_train, axes=(0, 2, 1))
        X_test = np.transpose(X_test, axes=(0, 2, 1))

        all_acc = []

        for e in tqdm.tqdm(range(1, 31)):

            if e == 1:
                X_train_c = X_train.copy()
                X_test_c = X_test.copy()
                y_train_c = y_train.copy()
                y_test_c = y_test.copy()
            else:
                X_train_c, y_train_c, X_test_c, y_test_c = stratified_resample_data(
                    X_train, y_train, X_test, y_test, random_state=e
                )

            # Scale to [0, 1]
            min_t = np.min(X_train_c, axis=0)
            max_t = np.max(X_train_c, axis=0)
            X_train_c = (X_train_c - min_t) / (max_t - min_t)
            X_test_c = (X_test_c - min_t) / (max_t - min_t)

            # Stratified KFold
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

            def objective(trial):
                _, seq_len, input_dim = X_train_c.shape

                internal_model = NClass(
                    input_dim=input_dim, output_dim=1, memory_length=0, device=device
                )

                activation = DTWActivation(
                    seq_len=seq_len,
                    input_dim=input_dim,
                    output_dim=1,
                    alpha=trial.suggest_float("alpha", 0.1, 0.9),
                    neighbor_rate=0,
                    device=device,
                )

                trainer = DTWTrainer(
                    activation=activation,
                    internal_model=internal_model,
                    R=[
                        trial.suggest_float(f"R{i}", 0.001, 0.4)
                        for i in range(input_dim)
                    ],
                    bad_th=trial.suggest_float("bad_th", 0.001, 0.9),
                    n_epochs=5,
                    learning_rules=[
                        IfNoActivatedAndNoNeighbors(),
                        IfNoActivated(),
                        IfActivated(),
                        SimpleDestroy(10000),
                    ],
                    device=device,
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

            study = optuna.create_study(
                sampler=TPESampler(seed=seed), direction="maximize"
            )
            study.optimize(objective, n_trials=n_trials, callbacks=[callback])

            # Evaluate
            best_params = study.best_params

            _, seq_len, input_dim = X_train_c.shape

            internal_model = NClass(
                input_dim=input_dim, output_dim=1, memory_length=0, device=device
            )

            activation = DTWActivation(
                seq_len=seq_len,
                input_dim=input_dim,
                output_dim=1,
                alpha=best_params["alpha"],
                neighbor_rate=0,
                device=device,
            )

            trainer = DTWTrainer(
                activation=activation,
                internal_model=internal_model,
                R=[best_params[f"R{i}"] for i in range(input_dim)],
                bad_th=best_params["bad_th"],
                n_epochs=5,
                learning_rules=[
                    IfNoActivatedAndNoNeighbors(),
                    IfNoActivated(),
                    IfActivated(),
                    SimpleDestroy(10000),
                ],
                device=device,
            )

            dataset = DataBuffer(X_train_c, y_train_c, device=device)

            trainer.fit(dataset)

            y_pred = []
            for batch in torch.tensor(
                X_test_c, dtype=torch.float32, device=device
            ).split(32):
                y_pred += trainer.predict(batch).cpu().tolist()

            all_acc.append(accuracy_score(y_pred, y_test_c))

        all_acc = np.array(all_acc)
        print(f"Accuracy: {np.mean(all_acc)}")
        print(f"Std: {np.std(all_acc)}")
