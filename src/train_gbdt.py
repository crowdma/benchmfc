import json
import random
from datetime import datetime
from timeit import default_timer as timer

import numpy as np
import pyrootutils
import typer
import wandb
from loguru import logger
from rich.progress import track
from sklearn.metrics import classification_report, top_k_accuracy_score
from wandb.lightgbm import log_summary, wandb_callback

ROOT = pyrootutils.setup_root(
    __file__,
    indicator=".project-root",
    pythonpath=True,
)

from src.datasets.mfc import MFC_LOADER
from src.models.gbdt import GBDTClassifier

app = typer.Typer()


def pprint(data):
    return json.dumps(data, indent=2)


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)


@app.command()
def main(
    data_name: str = None,
    feature_name: str = "feature-ember-npy",
    task_group: str = "gbdt-ember",
    pack_ratio: float = 0.0,
    train_size: float = 0.6,
    val_size: float = 0.2,
    test_size: float = 0.2,
    boosting: str = "gbdt",
    objective: str = "multiclass",
    num_class: int = 8,
    metric: str = "multi_logloss",
    num_iterations: int = 1_000,
    learning_rate: float = 0.05,
    num_leaves: int = 2048,
    max_depth: int = 15,
    min_data_in_leaf: int = 50,
    feature_fraction: float = 0.5,
    verbosity: int = -1,
    device: str = "cpu",
    num_threads: int = 20,
    seed: int = 42,
    do_wandb: bool = False,
    project: str = "lab-benchmfc",
):
    seed_everything(seed)
    # gbdt_config
    gbdt_params = {
        "boosting": boosting,
        "objective": objective,
        "num_class": num_class,
        "metric": metric,
        "num_iterations": num_iterations,
        "learning_rate": learning_rate,
        "num_leaves": num_leaves,
        "max_depth": max_depth,
        "min_data_in_leaf": min_data_in_leaf,
        "feature_fraction": feature_fraction,
        "device": device,
        "num_threads": num_threads,
        "verbosity": verbosity,
    }
    config = locals()

    gbdt = GBDTClassifier(**gbdt_params)
    # time
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # log_dir
    task_name = f"{task_group}-{data_name}-{pack_ratio}"
    log_dir = ROOT / f"logs/{task_name}/runs/{now}"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.add(f"{log_dir}/train.log", level="INFO")

    logger.info(f"[-] Global seed = {seed}")
    logger.info(f"[-] Config: {pprint(config)}")
    logger.info(f"[-] GBDT Params: {pprint(gbdt_params)}")

    start = timer()
    # prepare data
    X_train, X_val, X_test, y_train, y_val, y_test = MFC_LOADER[data_name].load(
        feature=feature_name,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        pack_ratio=pack_ratio,
    )
    X_train = [
        np.load(i)
        for i in track(X_train, total=len(X_train), description="Loading train...")
    ]
    X_test = [
        np.load(i)
        for i in track(X_test, total=len(X_test), description="Loading test...")
    ]

    X_train = np.vstack(X_train)
    y_train = np.array(y_train)
    X_test = np.vstack(X_test)
    y_test = np.array(y_test)
    # train
    name = task_name
    if do_wandb:
        wandb.login()
        wandb.init(
            project=project, name=name, group=task_group, config=config, dir=log_dir
        )
        gbdt.fit(X_train, y_train, callbacks=[wandb_callback()])
        log_summary(gbdt.model, feature_importance=True)
    else:
        gbdt.fit(X_train, y_train)
    # save model
    model_file = log_dir / f"{task_name}.lbg"
    logger.info(f"[-] save model: {model_file}")
    gbdt.model.save_model(model_file)
    # test
    predict = gbdt.predict(X_test)
    acc = top_k_accuracy_score(y_true=y_test, y_score=predict, k=1)
    logger.info(f"[*] Top-1 accuracy: {acc}")
    if do_wandb:
        wandb.log({"test/acc": acc})
        wandb.finish()

    predict = [np.argmax(i) for i in predict]
    result = classification_report(
        y_true=y_test, y_pred=predict, digits=4, output_dict=True
    )
    logger.info(f"[*] Classification_report (macro avg): {pprint(result['macro avg'])}")

    end = timer()
    logger.info(f"[-] timecost: {end - start} s")


if __name__ == "__main__":
    app()
