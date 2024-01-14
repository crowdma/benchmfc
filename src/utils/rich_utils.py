from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import rich
import rich.syntax
import rich.tree
from hydra.core.hydra_config import HydraConfig
from lightning.pytorch.utilities import rank_zero_only
from omegaconf import DictConfig, OmegaConf, open_dict
from rich.prompt import Prompt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)

from src.datasets.ember import EmberDataModule
from src.utils import pylogger

log = pylogger.get_pylogger(__name__)


@rank_zero_only
def print_config_tree(
    cfg: DictConfig,
    print_order: Sequence[str] = (
        "data",
        "model",
        "callbacks",
        "logger",
        "trainer",
        "paths",
        "extras",
    ),
    resolve: bool = False,
    save_to_file: bool = False,
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.
        print_order (Sequence[str], optional): Determines in what order config components are printed.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
        save_to_file (bool, optional): Whether to export config to the hydra output folder.
    """

    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    queue = []

    # add fields from `print_order` to queue
    for field in print_order:
        queue.append(field) if field in cfg else log.warning(
            f"Field '{field}' not found in config. Skipping '{field}' config printing..."
        )

    # add all the other fields to queue (not specified in `print_order`)
    for field in cfg:
        if field not in queue:
            queue.append(field)

    # generate config tree from queue
    for field in queue:
        branch = tree.add(field, style=style, guide_style=style)

        config_group = cfg[field]
        if isinstance(config_group, DictConfig):
            branch_content = OmegaConf.to_yaml(config_group, resolve=resolve)
        else:
            branch_content = str(config_group)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    # print config tree
    rich.print(tree)

    # save config tree to file
    if save_to_file:
        with open(Path(cfg.paths.output_dir, "config_tree.log"), "w") as file:
            rich.print(tree, file=file)


@rank_zero_only
def enforce_tags(cfg: DictConfig, save_to_file: bool = False) -> None:
    """Prompts user to input tags from command line if no tags are provided in config."""

    if not cfg.get("tags"):
        if "id" in HydraConfig().cfg.hydra.job:
            raise ValueError("Specify tags before launching a multirun!")

        log.warning("No tags provided in config. Prompting user to input tags...")
        tags = Prompt.ask("Enter a list of comma separated tags", default="dev")
        tags = [t.strip() for t in tags.split(",") if t != ""]

        with open_dict(cfg):
            cfg.tags = tags

        log.info(f"Tags: {cfg.tags}")

    if save_to_file:
        tags = [str(i) for i in cfg.tags]
        with open(Path(cfg.paths.output_dir, "tags.log"), "w") as file:
            rich.print(tags, file=file)


@rank_zero_only
def log_test_results(cfg: DictConfig, y_true: list, y_pred: list) -> None:
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # classification report
    cm = confusion_matrix(y_true, y_pred)
    cr = classification_report(y_true, y_pred, digits=4, zero_division=0)
    rich.print("Test Classification Report:")
    rich.print(cr)

    test_path = Path(cfg.paths.output_dir) / "test"
    test_path.mkdir(parents=True, exist_ok=True)

    cm_fig = ConfusionMatrixDisplay(cm).plot().figure_
    cm_fig.savefig(test_path / "confusion_matrix.png")
    np.savetxt(test_path / "confusion_matrix.txt", cm, fmt="%d")

    with open(test_path / "classification_report.log", "w") as file:
        rich.print(cr, file=file)


def plot_features(
    X: np.array,
    labels: np.array,
    max_length: int = 4096,
) -> plt.figure:
    fig, ax = plt.subplots()
    for i, j in zip(X, labels):
        ax.plot(i[:max_length], label=j)
    ax.set_title("Example Features of Classes")
    ax.legend(title="class")
    ax.set_xlabel("id")
    ax.set_ylabel("value")
    return fig


@rank_zero_only
def log_data_summary(cfg: DictConfig, data_module: EmberDataModule) -> None:
    data_summary = data_module.summary()
    save_path = Path(cfg.paths.output_dir) / "data"
    save_path.mkdir(parents=True, exist_ok=True)

    with open(save_path / "data_summary.log", "w") as file:
        rich.print(data_summary, file=file)

    data_loader = data_module.test_dataloader()
    X, y = next(iter(data_loader))

    if isinstance(y, list):
        y = y[-1]
    if isinstance(X, list):
        X = X[0]
    X, y = X.numpy(), y.numpy()

    batch_size = X.shape[0]
    X = X.reshape(batch_size, -1)

    labels, indices = np.unique(y, return_index=True)
    X = X[indices]
    fig = plot_features(X, labels)

    fig.savefig(save_path / "data_example.png")
