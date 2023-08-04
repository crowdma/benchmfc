from pathlib import Path

import numpy as np
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from .mfc import MFC_LOADER, Feature


def read_bytes(file_path: Path, first_n_byte: int) -> np.array:
    with open(file_path, "rb") as f:
        # index 0 will be special padding index
        data = [i + 1 for i in f.read()[:first_n_byte]]
        data = data + [0] * (first_n_byte - len(data))
        data = np.array(data, dtype=np.int32)
        # data = np.array(data, dtype=np.float32)
        return data


class BytesDataset(Dataset):
    """Ember Feature in-memory Dataset"""

    def __init__(self, X: list[Path], y: np.array, first_n_byte: int):
        self.X = X
        self.y = y
        self.first_n_byte = first_n_byte

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int) -> tuple:
        data = read_bytes(self.X[index], self.first_n_byte)
        target = self.y[index]
        return data, target


class BytesDataModule(LightningDataModule):
    """LightningDataModule for Ember Feature dataset.

    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    def __init__(
        self,
        data_name: str,
        train_size: float = 0.6,
        val_size: float = 0.2,
        test_size: float = 0.2,
        batch_size: int = 32,
        num_workers: int = 16,
        pack_ratio: float = 0.0,
        first_n_byte: int = 2**20,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters()

        self.data_train: Dataset = None
        self.data_val: Dataset = None
        self.data_test: Dataset = None

    def setup(self, stage: str = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning when doing `trainer.fit()` and `trainer.test()`,
        so be careful not to execute the random split twice! The `stage` can be used to
        differentiate whether it's called before trainer.fit()` or `trainer.test()`.
        """

        # load datasets only if they're not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            loader = MFC_LOADER[self.hparams.data_name]
            (X_train, X_val, X_test, y_train, y_val, y_test) = loader.load(
                feature=Feature.SAMPLES,
                train_size=self.hparams.train_size,
                val_size=self.hparams.val_size,
                test_size=self.hparams.test_size,
                pack_ratio=self.hparams.pack_ratio,
            )
            first_n_byte = self.hparams.first_n_byte
            self.data_train = BytesDataset(X_train, y_train, first_n_byte)
            self.data_val = BytesDataset(X_val, y_val, first_n_byte)
            self.data_test = BytesDataset(X_test, y_test, first_n_byte)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
        )
