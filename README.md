# LUMEN
Codes for "Prediction of Pulmonary Dysfunction from Chest CT Scans via a Large-language-model Guided Multimodal Framework"

## Contents
- [Use Terms](#use-terms)
- [Introduction](#introduction)
- [Model](#model)
- [Deployment](#deployment)
- [Requirements](#requirements)
- [Installation](#installation)


## Use Terms

### Intellectual Property and Rights Notice
All content within this repository, including but not limited to source code, models, algorithms, data, and documentation, are subject to applicable intellectual property laws. The rights to this project are reserved by the project's author(s) or the rightful patent holder(s).

### Limitations on Commercial Use
This repository's contents, protected by patent, are solely for personal learning and research purposes, and are not for commercial use. Any organizations or individuals must not use any part of this project for commercial purposes without explicit written permission from the author(s) or rightful patent holder(s). Violations of this restriction will result in legal action.

### Terms for Personal Learning and Academic Research
Individual users are permitted to use this repository for learning and research purposes, provided that they abide by applicable laws. Should you utilize this project in your research, please cite our work as follows:

> Zhou, J., Hang, Y., Bin, H. et al. Prediction of Pulmonary Dysfunction from Chest CT Scans via a Large-language-model Guided Multimodal Framework.

## Introduction
Pulmonary function testing (PFT) is the clinical standard for diagnosing chronic respiratory diseases, yet its utility is often limited by accessibility constraints and patient compliance. While computed tomography (CT) captures structural lung abnormalities associated with pulmonary dysfunction, existing deep-learning diagnostic models typically rely solely on imaging or struggle to effectively integrate unstructured clinical text. 

Here, we present LUMEN, a unified multimodal artificial intelligence framework for the automated prediction of both obstructive and restrictive ventilatory impairments. The framework leverages a large language model with medical chain-of-thought prompting to distill heterogeneous CT reports into structured reasoning features, which are then fused with CT image representations via a bidirectional cross-attention mechanism. To mitigate class imbalance and ensure the reliability of the generated reasoning text, we implemented a confidence-aware three-stage training strategy. Validated on 5,401 patients from three medical centers, LUMEN accurately identified high-risk populations based on multimodal clinical characteristics (i.e., CT images, CT reports, etc), consistently outperforming state-of-the-art unimodal and multimodal baselines.
## Model

![LUMEN-CATS.png]
## Deployment

## Requirements
