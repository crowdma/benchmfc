"""Malware Family Classification Data"""
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

MFC_ROOT = Path(os.getenv("MFC_ROOT"))

PACKERS = ["upx", "mpress", "aes"]


class Feature:
    EMBER_NUMPY = "feature-ember-npy"
    SAMPLES = "samples"


FEATURE_SUFFIX = {
    Feature.SAMPLES: "",
    Feature.EMBER_NUMPY: "ember.npy",
}


class Group:
    MALICIOUS = "malicious"
    MALICIOUS_UNSEEN = "malicious-unseen"
    MALICIOUS_EVOLVING = "malicious-evolving"
    MALICIOUS_UPX = "malicious-upx"
    MALICIOUS_MPRESS = "malicious-mpress"
    MALICIOUS_AES = "malicious-aes"


class FeatureLoader:
    name = "feature_loader"
    dtype = "float32"

    def __call__(self, file_path: str) -> np.ndarray:
        raise NotImplementedError


class NumpyLoader(FeatureLoader):
    name = "numpy"
    dtype = "float32"

    def __call__(self, file_path: str) -> np.ndarray:
        data: np.ndarray = np.load(file_path)
        return data.astype(np.float32)


class MalconvByteLoader(FeatureLoader):
    name = "malconv_byte"
    dtype = "int32"

    def __init__(self, first_n_byte: int = 2**20) -> None:
        self.first_n_byte = first_n_byte

    def __call__(self, file_path: str) -> np.ndarray:
        with open(file_path, "rb") as f:
            # index 0 will be special padding index
            data = [i + 1 for i in f.read()[: self.first_n_byte]]
            data = data + [0] * (self.first_n_byte - len(data))
            return np.array(data).astype(np.intc)


def class_counter(data: list[str]) -> dict:
    return dict(sorted(Counter(data).items(), key=lambda x: x[1], reverse=True))


class MFCSample:
    """MFC Sample is orginazied by:
    ```
    root/samples/<data-group>/<tag>/xxx

    >>> For example:
    MFC
    ├── samples
    │  ├── malicious
    │  │  ├── fareit
    │  │  ├── gandcrab
    │  │  ├── hotbar
    │  │  ├── parite
    │  │  ├── simda
    │  │  ├── upatre
    │  │  ├── yuner
    │  │  └── zbot
    ....
    ```
    """

    root: str = MFC_ROOT
    group: str = Group.MALICIOUS

    def get(
        self,
        group: str = None,
        root: str = None,
    ) -> tuple[list[str], list[str]]:
        group = group or self.group
        root = root or self.root
        data_path = Path(root) / "samples"

        X = []
        y = []
        for r, _, files in os.walk(data_path / group):
            for f in files:
                file_path = Path(r, f)
                X.append(file_path.name)
                y.append(file_path.parent.name)

        if len(X) == 0:
            raise ValueError(f"Empty: {root}/samples/{group}")

        return X, y


class MFCLoader:
    root: Path = MFC_ROOT
    group: str = Group.MALICIOUS
    class_map: dict = {
        "fareit": 0,
        "gandcrab": 1,
        "hotbar": 2,
        "parite": 3,
        "simda": 4,
        "upatre": 5,
        "yuner": 6,
        "zbot": 7,
    }
    packers: list[str] = None

    def __init__(self):
        self.X_train = None
        self.X_test = None
        self.X_val = None
        self.y_train = None
        self.y_test = None
        self.y_val = None
        self.feature = None
        self.pack_ratio = None
        self.load_fn: FeatureLoader = None

    def get_path(self, X: list[str], y: list[str]) -> tuple[list[Path], list[int]]:
        X_path = []
        y_id = []

        for name, family in zip(X, y):
            if "_" in name:
                group = "-".join([self.group] + name.split("_")[1:])
            else:
                group = self.group
            suffix = FEATURE_SUFFIX[self.feature]
            if suffix:
                name = f"{name}_{FEATURE_SUFFIX[self.feature]}"
            p = self.root / self.feature / group / family / name
            if p.exists():
                X_path.append(p)
                y_id.append(self.class_map[family])
        assert len(X_path) > 0
        return X_path, y_id

    def pack(
        self,
        X: list[str],
        y: list[int],
        pack_ratio: float,
    ) -> tuple[list[str], list[int]]:
        if pack_ratio == 1.0:
            X_packed, y_packed = X, y
            X_unpacked, y_unpacked = [], []
        else:
            X_packed, X_unpacked, y_packed, y_unpacked = train_test_split(
                X, y, train_size=pack_ratio, stratify=y, random_state=42
            )
        num = len(X_packed)
        packers = self.packers
        m = len(packers)
        n = num // m
        packed = []
        for i, j in enumerate(range(0, num, n)):
            packed.extend(
                [f"{k}_{packers[i%m]}" for k in X_packed[j : min(num, j + n)]]
            )
        return packed + X_unpacked, y_packed + y_unpacked

    def setup(
        self,
        feature: str = None,
        pack_ratio: float = None,
        train_size: float = 0.6,
        val_size: float = 0.2,
        test_size: float = 0.2,
    ) -> tuple[list[Path], list[Path], list[Path], list[int], list[int], list[int]]:
        """
        Returns
        -------
        X_train, X_val, X_test, y_train, y_val, y_test
        """
        assert sum([train_size, val_size, test_size]) == 1.0

        group = self.group
        root = self.root

        self.feature = feature
        self.pack_ratio = pack_ratio

        if feature == Feature.EMBER_NUMPY:
            self.load_fn = NumpyLoader()
        elif feature == Feature.SAMPLES:
            self.load_fn = MalconvByteLoader()
        else:
            raise ValueError(f"Unknown feature: {feature}")

        X, y = MFCSample().get(group, root)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=train_size, stratify=y, random_state=42
        )
        # # for 40% test samples
        # new_size = test_size
        new_size = test_size / (test_size + val_size)
        X_test, X_val, y_test, y_val = train_test_split(
            X_test, y_test, train_size=new_size, stratify=y_test, random_state=42
        )

        # pack
        if pack_ratio is None:
            pack_ratio = self.pack_ratio
        if 0.1 <= pack_ratio <= 1.0:
            X_train, y_train = self.pack(X_train, y_train, pack_ratio)
            X_test, y_test = self.pack(X_test, y_test, pack_ratio)
            X_val, y_val = self.pack(X_val, y_val, pack_ratio)

        # path
        X_train, y_train = self.get_path(X_train, y_train)
        X_test, y_test = self.get_path(X_test, y_test)
        X_val, y_val = self.get_path(X_val, y_val)

        self.X_train = X_train
        self.X_test = X_test
        self.X_val = X_val
        self.y_train = y_train
        self.y_test = y_test
        self.y_val = y_val

        return (X_train, X_val, X_test, y_train, y_val, y_test)

    def is_packed(self, x: Path) -> bool:
        return any([i in x.name for i in PACKERS])

    def get_packed_ratio(self, X: list[Path]) -> float:
        ratio = sum([self.is_packed(i) for i in X]) / len(X)
        return round(ratio, 2)

    def get_packer_dist(self, X: list[list[Path]]) -> dict[str, int]:
        packers = defaultdict(int)
        for f in X:
            hit = False
            for p in PACKERS:
                if p in f.name:
                    hit = True
                    packers[p] += 1
            if not hit:
                packers["none"] += 1
        return dict(packers)

    def load(self, x: Path) -> np.ndarray:
        return self.load_fn(x)

    def summary(self) -> dict:
        X_train, X_val, X_test = self.X_train, self.X_val, self.X_test
        y_train, y_val, y_test = self.y_train, self.y_val, self.y_test
        num_train, num_val, num_test = len(X_train), len(X_val), len(X_test)

        ratio_train = self.get_packed_ratio(X_train)
        ratio_val = self.get_packed_ratio(X_val)
        ratio_test = self.get_packed_ratio(X_test)

        packers_train = self.get_packer_dist(X_train)
        packers_val = self.get_packer_dist(X_val)
        packers_test = self.get_packer_dist(X_test)

        def data_class(y: list[int]) -> dict[str, int]:
            return dict(sorted(Counter(y).items()))

        # data
        data = {
            "train": {
                "total": num_train,
                "packer": packers_train,
                "packed_ratio": ratio_train,
                "class": data_class(y_train),
            },
            "val": {
                "total": num_val,
                "packer": packers_val,
                "packed_ratio": ratio_val,
                "class": data_class(y_val),
            },
            "test": {
                "total": num_test,
                "packer": packers_test,
                "packed_ratio": ratio_test,
                "class": data_class(y_test),
            },
        }
        # feature
        x_data = self.load(X_train[0])
        x_path = str(X_train[0].relative_to(self.root))
        feature = {
            "names": self.feature,
            "loader": self.load_fn.name,
            "dtype": self.load_fn.dtype,
            "example": x_path,
            "dimension": len(x_data),
        }
        return {"data": data, "feature": feature}


class MFC(MFCLoader):
    group: str = Group.MALICIOUS
    class_map: dict = {
        "fareit": 0,
        "gandcrab": 1,
        "hotbar": 2,
        "parite": 3,
        "simda": 4,
        "upatre": 5,
        "yuner": 6,
        "zbot": 7,
    }


class MFCEvolving(MFC):
    group: str = Group.MALICIOUS_EVOLVING


class MFCUnseen(MFCLoader):
    group: str = Group.MALICIOUS_UNSEEN
    class_map: dict = {
        "hupigon": 0,
        "imali": 1,
        "lydra": 2,
        "onlinegames": 3,
        "virut": 4,
        "vobfus": 5,
        "wannacry": 6,
        "zlob": 7,
    }


class MFCPacking(MFC):
    packers: list[str] = ["upx", "mpress", "aes"]
    pack_ratio: float = 1.0


class MFCAes(MFC):
    packers: list[str] = ["aes"]
    pack_ratio: float = 1.0


MFC_LOADER: dict[str, MFCLoader] = {
    "MFC": MFC(),
    "MFCAes": MFCAes(),
    "MFCEvolving": MFCEvolving(),
    "MFCPacking": MFCPacking(),
    "MFCUnseen": MFCUnseen(),
}


def get_dataloader(name: str) -> MFCLoader:
    return MFC_LOADER[name]


if __name__ == "__main__":
    import rich

    mfc = MFCPacking()
    mfc.setup()
    rich.print(mfc.summary())
