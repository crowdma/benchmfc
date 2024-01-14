import json
import random
from pathlib import Path

import numpy as np
import pyrootutils
import rich
import typer
from rich.progress import track
from sklearn.metrics import classification_report

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
    ckpt_path: str = None,
    feature_name: str = "feature-ember-npy",
    pack_ratio: float = 0.0,
    train_size: float = 0.6,
    val_size: float = 0.2,
    test_size: float = 0.2,
    seed: int = 42,
):
    seed_everything(seed)
    gbdt = GBDTClassifier()
    gbdt.load(ckpt_path)

    ckpt_path = Path(ckpt_path)
    log_dir = ckpt_path.parent / f"{data_name}"
    log_dir.mkdir(parents=True, exist_ok=True)

    # prepare data
    mfc = MFC_LOADER[data_name]
    mfc.setup(
        feature=feature_name,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        pack_ratio=pack_ratio,
    )
    rich.print(mfc.summary())
    with open(log_dir / "data_summary.log", "w") as file:
        rich.print(mfc.summary(), file=file)

    _, _, X_test, _, _, y_test = (
        mfc.X_train,
        mfc.X_val,
        mfc.X_test,
        mfc.y_train,
        mfc.y_val,
        mfc.y_test,
    )
    X_test = [
        np.load(i)
        for i in track(X_test, total=len(X_test), description="Loading test...")
    ]
    X_test = np.vstack(X_test)
    y_test = np.array(y_test)

    # predict
    predict = gbdt.predict(X_test)
    predict = [np.argmax(i) for i in predict]

    result = classification_report(y_true=y_test, y_pred=predict, digits=4)
    rich.print(f"Test Report: {result}")
    with open(log_dir / "test_results.log", "w") as file:
        rich.print(result, file=file)


if __name__ == "__main__":
    app()
