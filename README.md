
<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/crowdma/benchmfc">
  </a>

  <h3 align="center">BenchMFC</h3>

  <p align="center">
    A Benchmark Dataset for Trustworthy Malware Family Classification under Concept Drift
  </p>
</div>



<!-- ABOUT THE PROJECT -->
## Abstract

Concept drift poses a critical challenge in deploying machine learning models to mitigate practical malware threats. It refers to the phenomenon that the distribution of test data changes over time, gradually deviating from the original training data and degrading model performance. A promising direction for addressing concept drift is to detect drift samples and then retrain the model. However, this field currently lacks a unified, well-curated, and comprehensive benchmark, which often leads to unfair comparisons and inconclusive outcomes. To improve the evaluation and advance further, this paper presents a new Benchmark dataset for trustworthy Malware Family Classification (BenchMFC), which includes 223K samples of 526 families that evolve over years. BenchMFC provides clear family, packer, and timestamp tags for each sample, it thus can support research on three types of malware concept drift: 1) unseen families, 2) packed families, and 3) evolved families. To collect unpacked family samples from large-scale candidates, we introduce a novel crowdsourcing malware annotation pipeline, which unifies packing detection and family annotation as a consensus inference problem to prevent costly packing detection. Moreover, we provide two case studies to illustrate the application of BenchMFC in 1) concept drift detection and 2) model retraining. The first case demonstrates the impact of three types of malware concept drift and compares nine notable concept drift detectors. The results show that existing detectors have their own advantages in dealing with different types of malware concept drift, and there is still room for improvement in malware concept drift detection. The second case explores how static feature-based machine learning operates on packed samples when retraining a model. The experiments illustrate that packers do preserve some kind of signals that appear to be “effective” for machine learning models, but the robustness of these signals requires further research. BenchMFC has been released to the community at https://github.com/crowdma/benchmfc.


## Reference
This paper has been accepted by Computers & Security:
```
@article{jiang_2024,
title = {BenchMFC: A benchmark dataset for trustworthy malware family classification under concept drift},
author = {Yongkang Jiang and Gaolei Li and Shenghong Li and Ying Guo},
journal = {Computers & Security},
volume = {139},
pages = {103706},
year = {2024},
}
```

<!-- GETTING STARTED -->
## Dataset

### Size

```
├── benchmfc_meta.csv (Metadata file for the dataset ~17M)
├── benchmfc.tar.gz (Samples ~83G)
└── mfc (Experimental data used in the paper)
    ├── mfc_features.tar.gz (Ember features ~39M)
    ├── mfc_meta.csv (Metadata file ~1M)
    └── mfc_samples.tar.gz (Samples ~7G)
```

### Download
Please visit this [link](about/download.md) for more details.


## Getting Started

### Installation

- Run the following commands:
   ```sh
   # python = "<3.10 >=3.9"
   git clone https://github.com/crowdma/benchmfc.git
   cd benchmfc
   pip install -r requirements.txt
   ```

<!-- USAGE EXAMPLES -->
## Usage Examples

- Env

```sh
export MFC_ROOT=<path>/<to>/<MFC>
# MFC structure
├── feature-ember-npy
│   ├── malicious
│   ├── malicious-unseen
│   ├── malicious-evolving
│   ├── malicious-aes
│   ├── malicious-mpress
│   └── malicious-upx
└── samples
    ├── malicious
    ├── malicious-unseen
    ├── malicious-evolving
    ├── malicious-aes
    ├── malicious-mpress
    └── malicious-upx
```

- Train
```sh
/bin/bash scripts/train_mlp_ember.sh
```

- Test
```sh
/bin/bash scripts/test_mlp_ember.sh
```
- Detect Drift
```sh
/bin/bash scripts/detect_mlp_ember_drift.sh
```

## Issues

Please visit this [link](about/issues.md) for known issues.


<!-- LICENSE -->
## License

Distributed under the MIT License.