"""detect concept drift on mlp-ember"""
import os
import random

import numpy as np
import pandas as pd
import pyrootutils
import torch
import typer
from loguru import logger
from pytorch_ood.detector import (
    ODIN,
    EnergyBased,
    Entropy,
    KLMatching,
    Mahalanobis,
    MaxLogit,
    MaxSoftmax,
    ViM,
)
from pytorch_ood.utils import OODMetrics
from torch.utils.data import ConcatDataset, DataLoader, Dataset

ROOT = pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.datasets.ember import EmberDataModule
from src.models.mlp import MLP

app = typer.Typer(add_completion=False)


class CDDataset(Dataset):
    def __init__(self, X: list[np.array], y: list[int]):
        self.X = X
        self.y = y

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int) -> tuple:
        data, target = self.X[index], self.y[index]
        return data, target


def seed_everything(seed: int):
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@app.command()
def main(
    model_file: str = None,
    data_name: str = None,
    pack_ratio: float = 0.0,
    device: str = "cuda:0",
    seed: int = 42,
):
    # seed
    logger.info(f"Seedeverthing with {seed}")
    seed_everything(seed)
    # load model
    logger.info(f"Loading model from {model_file}")
    model = MLP()
    model.load_state_dict(torch.load(ROOT / model_file))
    model = model.eval().to(device)
    last_layer = model.model[-1]

    # load training data
    logger.info("Loading ID data MFC")
    ID = EmberDataModule(data_name="MFC")
    ID.setup()
    ID_train_loader = ID.train_dataloader()
    ID_test = ID.data_test

    # hit CD data
    logger.info(f"Hiting CD from {data_name}")
    if data_name in ["MFCUnseen", "MFCUnseenPacking"]:
        CD = EmberDataModule(data_name=data_name, pack_ratio=pack_ratio)
        CD.setup()
        CD_test = CD.data_test
        CD_test.y = [-1 for _ in CD_test.y]
    elif data_name in ["MFCPacking", "MFCEvolving"]:
        len(ID_test)
        # find cd data
        CD = EmberDataModule(data_name=data_name, pack_ratio=pack_ratio)
        CD.setup()
        CD_train_loader = CD.train_dataloader()
        CD_X = []
        with torch.no_grad():
            for x, y in CD_train_loader:
                logits = model(x.to(device))
                preds = torch.argmax(logits, dim=1).tolist()
                for i, p in enumerate(preds):
                    if p != y[i]:
                        CD_X.append(x[i].detach().numpy())
        CD_X = CD_X[: len(ID_test)]
        CD_y = [-1 for _ in CD_X]
        CD_test = CDDataset(CD_X, CD_y)
    else:
        logger.error(f"Unknown data_name {data_name}")
        raise typer.Exit()

    # pdb.set_trace()
    assert len(CD_test)
    logger.info(f"TestData | ID: {len(ID_test)}, CD: {len(CD_test)}")
    # concatenate ID and CD data
    test_data = ConcatDataset([ID_test, CD_test])
    test_loader = DataLoader(test_data, batch_size=32, shuffle=True)

    # create detector
    std = [1]
    logger.info("Creating detectors")
    detectors = {}
    detectors["MaxSoftmax"] = MaxSoftmax(model)
    detectors["ODIN"] = ODIN(model, norm_std=std, eps=0.002)
    detectors["Mahalanobis"] = Mahalanobis(model.features, norm_std=std, eps=0.002)
    detectors["EnergyBased"] = EnergyBased(model)
    detectors["Entropy"] = Entropy(model)
    detectors["MaxLogit"] = MaxLogit(model)
    detectors["KLMatching"] = KLMatching(model)
    detectors["ViM"] = ViM(model.features, d=64, w=last_layer.weight, b=last_layer.bias)

    # fit detectors to training data (some require this, some do not)
    logger.info(f"> Fitting {len(detectors)} detectors")
    for name, detector in detectors.items():
        logger.info(f"--> Fitting {name}")
        detector.fit(ID_train_loader, device=device)

    print(
        f"STAGE 3: Evaluating {len(detectors)} detectors on {data_name} concept drifts."
    )
    results = []

    with torch.no_grad():
        for detector_name, detector in detectors.items():
            print(f"> Evaluating {detector_name}")
            metrics = OODMetrics()
            for x, y in test_loader:
                metrics.update(detector(x.to(device)), y.to(device))

            r = {"Detector": detector_name}
            d = {k: round(v * 100, 2) for k, v in metrics.compute().items()}
            r.update(d)
            results.append(r)

    df = pd.DataFrame(
        results, columns=["Detector", "AUROC", "FPR95TPR", "AUPR-IN", "AUPR-OUT"]
    )
    df.to_csv(ROOT / f"detect/mlp-ember-{data_name}.csv", index=False)
    mean_scores = df.groupby("Detector").mean()
    logger.info(mean_scores.sort_values("AUROC").to_csv(float_format="%.2f"))


if __name__ == "__main__":
    app()
