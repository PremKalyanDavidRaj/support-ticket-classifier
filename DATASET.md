# Dataset: BANKING77

Creator: PolyAI.

Source: https://github.com/PolyAI-LDN/task-specific-datasets

Dataset card: https://huggingface.co/datasets/PolyAI/banking77

License: Creative Commons Attribution 4.0 International.
https://creativecommons.org/licenses/by/4.0/

Reference: Casanueva et al. (2020),
"Efficient Intent Detection with Dual Sentence Encoders."
https://arxiv.org/abs/2003.04807

This project downloads the original training and test CSV files
without modifications. Download URLs and SHA-256 hashes are
recorded in reports/data_manifest.json.

The task is classification of banking customer-service queries
into 77 intents. The dataset does not contain support-team or
priority labels.

The original test split is reserved for final evaluation.
A validation split will be created from the training data.
