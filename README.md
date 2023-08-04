## BenchMFC Malware Dataset

### Abstract

Machine learning for malware family classification shows encouraging results, but open-world deployments suffer from performance degradation as malware authors often use packing and other evasion techniques. This phenomenon is known as concept drift, occurs when malware packs, evolves and becomes less and less like the original training data. A promising direction to address concept drift is classification with rejection, in which potentially misclassified samples are detected and isolated until they can be expertly analyzed. However, this field currently lacks a unified, well-curated, and comprehensive benchmark, which often leads to unfair comparisons and inconclusive outcomes. To improve the evaluation and advance further, this paper presents a new Benchmark dataset for trustworthy Malware Family Classification (BenchMFC) under concept drift, which includes 223K unpacked samples of 467 families that evolve over years. BenchMFC provides scripts to generate different packed samples, it can support research on three types of malware concept drift: 1) unseen families, 2) packed families, and 3) evolved families. To collect unpacked family samples from large-scale candidates, we introduce a novel crowdsourcing malware annotation pipeline, which unifies packing detection and family annotation as a consensus inference problem to prevent costly packing detection. Moreover, we provide two case studies on BenchMFC: 1) multi-type malware concept drift detection, and 2) multi-class malware family classification. The extensive experiments compare 9 notable concept drift detectors and explore how static feature-based machine learning operates on packed families, which illustrates the impact of malware concept drift, shows missing elements in existing methods, and discusses how BenchMFC can support future research.


### Download

To avoid misuse, please read and agree to the following conditions before sending us emails.

- Please email Yongkang (jiangyongkang@sjtu.edu.cn). Also, please include your Gmail address in the body so that I can add you to the google drive folder where the dataset is stored.
- Do not share the data with any others (except your co-authors for the project). We are happy to share with other researchers based upon their requests.
- Explain in a few sentences of your plan to do with these binaries. It should not be a precise plan.
- If you are in academia, contact us using your institution email and provide us a webpage registered at the university domain that contains your name and affiliation.
- If you are in research (industrial) labs, send us an email from your company’s email account and introduce yourself and company. In the email, please attach a justification letter (in PDF format) in official letterhead. The letter needs to state clearly the reasons why this dataset is being requested.

Please note that an email not following the conditions might be ignored. And we will keep the public list of organizations accessing these samples at the bottom.