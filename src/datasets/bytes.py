from pathlib import Path

import numpy as np
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from src import utils

from .mfc import Feature, MalconvByteLoader, get_dataloader

log = utils.get_pylogger(__name__)


class BytesDataset(Dataset):
    """Ember Feature in-memory Dataset"""

    def __init__(self, X: list[Path], y: np.array, load_fn: callable):
        self.X = X
        self.y = y
        self.load_fn = load_fn

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int) -> tuple:
        data = self.load_fn(self.X[index])
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

        self.mfc = get_dataloader(data_name)
        self.load_fn = MalconvByteLoader(first_n_byte=first_n_byte)

    def summary(self) -> dict:
        if self.mfc.X_train is None:
            self.mfc.setup()
        return self.mfc.summary()

    def setup(self, stage: str = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning when doing `trainer.fit()` and `trainer.test()`,
        so be careful not to execute the random split twice! The `stage` can be used to
        differentiate whether it's called before trainer.fit()` or `trainer.test()`.
        """

        # load datasets only if they're not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            mfc = self.mfc
            mfc.setup(
                feature=Feature.SAMPLES,
                pack_ratio=self.hparams.pack_ratio,
                train_size=self.hparams.train_size,
                val_size=self.hparams.val_size,
                test_size=self.hparams.test_size,
            )
            log.info(f"Summary: {self.summary()}")
            (X_train, X_val, X_test, y_train, y_val, y_test) = (
                mfc.X_train,
                mfc.X_val,
                mfc.X_test,
                mfc.y_train,
                mfc.y_val,
                mfc.y_test,
            )

            load_fn = self.load_fn
            self.data_train = BytesDataset(X_train, y_train, load_fn)
            self.data_val = BytesDataset(X_val, y_val, load_fn)
            self.data_test = BytesDataset(X_test, y_test, load_fn)

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
