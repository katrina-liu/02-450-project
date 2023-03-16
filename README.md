# Comparison of Active Learning Classification Methods on High-Volume Cancer Gene Expression Data 
02-450 Final Project

## Description
We used the GENT2 datasets to study whether active learning strategy can better predict cancer state of the samples.To provide a comprehensive basis for testing and comparing learning methods on cancer data, features were selected from a set of significantly mutated genes, and from a variety of mechanisms involved in cancer
We adopted baseline methods including Random Forest, Support Vector Classification with Standard Scaler, to compare with Active Learning Query Strategies inlcuding Uncertainty Sampling with Least Confidence Level, Query By Committee with Committee Members for Each Gene, Expected Error Reduction.

## Usage
The project is developed using a object-oriented framework-plugin structure, where the base framework learner file is at `src/BaseLearner.py` and active learning strategies stored under the folder `src/active_learners` inheriting from the base learner.
To run each active learning strategy or base learner, simply run `python <filename>`.

## Contribution
Katrina Liu
Cameron Miller
Qingyi Peng

## License
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
