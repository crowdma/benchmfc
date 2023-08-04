"""Malware Family Classification Data"""
import os
from collections import Counter
from pathlib import Path

from sklearn.model_selection import train_test_split

MFC_ROOT = Path(os.getenv("MFC_ROOT"))


class Feature:
    EMBER_NUMPY = "ember-feature-npy"
    BYTES_NUMPY = "bytes-feature-npy"
    SAMPLES = "samples"


FEATURE_SUFFIX = {
    Feature.SAMPLES: "",
    Feature.EMBER_NUMPY: "ember.npy",
    Feature.BYTES_NUMPY: "bytes.npy",
}


class Group:
    MALICIOUS = "malicious"
    MALICIOUS_UNSEEN = "malicious-unseen"
    MALICIOUS_EVOLVING = "malicious-evolving"
    MALICIOUS_UPX = "malicious-upx-packed"
    MALICIOUS_MPRESS = "malicious-mpress-packed"
    MALICIOUS_AES = "malicious-aes-packed"


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

    @classmethod
    def load(
        cls,
        group: str = None,
        root: str = None,
    ) -> tuple[list[str], list[str]]:
        group = group or cls.group
        root = root or cls.root
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


class MFCPath:
    root: str = MFC_ROOT
    feature: str = Feature.EMBER_NUMPY
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
    pack_ratio: float = None

    @classmethod
    def get_path(cls, X: list[str], y: list[str]) -> tuple[list[Path], list[int]]:
        X_path = []
        y_id = []
        for name, family in zip(X, y):
            if "_" in name:
                group = "-".join([cls.group] + name.split("_")[1:]) + "-packed"
            else:
                group = cls.group
            suffix = FEATURE_SUFFIX[cls.feature]
            if suffix:
                name = f"{name}_{FEATURE_SUFFIX[cls.feature]}"
            p = cls.root / cls.feature / group / family / name
            if p.exists():
                X_path.append(p)
                y_id.append(cls.class_map[family])
        assert len(X_path) > 0
        return X_path, y_id

    @classmethod
    def pack(
        cls,
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
        packers = cls.packers
        m = len(packers)
        n = num // m
        packed = []
        for i, j in enumerate(range(0, num, n)):
            packed.extend(
                [f"{k}_{packers[i%m]}" for k in X_packed[j : min(num, j + n)]]
            )
        return packed + X_unpacked, y_packed + y_unpacked

    @classmethod
    def load(
        cls,
        feature: str = None,
        group: str = None,
        root: str = None,
        train_size: float = 0.6,
        val_size: float = 0.2,
        test_size: float = 0.2,
        pack_ratio: float = None,
    ) -> tuple[list[Path], list[Path], list[Path], list[int], list[int], list[int]]:
        """
        Returns
        -------
        X_train, X_val, X_test, y_train, y_val, y_test
        """
        assert sum([train_size, val_size, test_size]) == 1.0

        feature = feature or cls.feature
        group = group or cls.group
        root = root or cls.root
        X, y = MFCSample.load(group, root)
        # y = [cls.class_map[i] for i in y]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=train_size, stratify=y, random_state=42
        )
        X_test, X_val, y_test, y_val = train_test_split(
            X_test, y_test, train_size=test_size, stratify=y_test, random_state=42
        )

        # pack
        if pack_ratio is None:
            pack_ratio = cls.pack_ratio
        if 0.1 <= pack_ratio <= 1.0:
            X_train, y_train = cls.pack(X_train, y_train, pack_ratio)
            X_test, y_test = cls.pack(X_test, y_test, pack_ratio)
            X_val, y_val = cls.pack(X_val, y_val, pack_ratio)

        # path
        X_train, y_train = cls.get_path(X_train, y_train)
        X_test, y_test = cls.get_path(X_test, y_test)
        X_val, y_val = cls.get_path(X_val, y_val)

        return (X_train, X_val, X_test, y_train, y_val, y_test)


class MFC(MFCPath):
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


class MFCUnseen(MFCPath):
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


class MFCUnseenPacking(MFCUnseen):
    packers: list[str] = ["upx", "mpress", "aes"]
    pack_ratio: float = 1.0


class MFCPacking(MFC):
    packers: list[str] = ["upx", "mpress", "aes"]
    pack_ratio: float = 1.0


class MFCAes(MFC):
    packers: list[str] = ["aes"]
    pack_ratio: float = 1.0


MFC_LOADER: dict[str, MFCPath] = {
    "MFC": MFC,
    "MFCAes": MFCAes,
    "MFCEvolving": MFCEvolving,
    "MFCPacking": MFCPacking,
    "MFCUnseen": MFCUnseen,
    "MFCUnseenPacking": MFCUnseenPacking,
}


def get_dataloader(name: str) -> MFCPath:
    return MFC_LOADER[name]


if __name__ == "__main__":
    # for _, loader in MFC_LOADER.items():
    #     loader.load()

    X_train, X_val, X_test, y_train, y_val, y_test = MFCAes.load(pack_ratio=0.9)
    inspect = []
    for i in X_train:
        d = i.name.split("_")
        if len(d) >= 3:
            inspect.append(d[-2])
        else:
            inspect.append("unpacked")
    print(Counter(inspect))
