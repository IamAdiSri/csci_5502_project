# Collaborative Filtering for Movie Reommendation

Comparing classical and neural recommendation algorithms on the Movielens dataset.

## Setup

1. Install Python >= 3.9.7
2. Run `pip install -r requirements.txt` to install required packages.
3. Download the datsets from the links below and unzip them into the root folder of the repository:
    - [Movielens 100k](https://grouplens.org/datasets/movielens/100k/)
    - [Movielens 1M](https://grouplens.org/datasets/movielens/1m/)

## About

This repository implements multiple versions of the following two recommendation algorithms:

1. Matrix Factorization ([Takacs et al., 2008](https://dl.acm.org/doi/10.1145/1454008.1454049))
2. Neural Matrix Factorization ([He et al., 2017](https://dl.acm.org/doi/10.1145/3038912.3052569))

The models have been implemented in Pytorch, and each model was trained and tested on the Movielens 100k dataset (complete implementations for using the Movielens 1M dataset are present in the repository, however they were not used in the experiments). The ratings in the dataset are converted from explicit to implicit, i.e. all user-item ratings in the dataset are considered to be positive samples and negative samples are sampled randomly from unrated items for each user.

Multiple combinations of objective functions (Root Mean Squared Error (RMSE), Binary Cross Entropy (BCE) and Bayesian Personalized Ranking (BPR)) were used to train the models. 

The SGD and Adam optimizers were used to optimize the models, and evaluation was conducted on the metrics Hit Ratio (HR) and Normalized Discounted Cumulative Gain (NDCG). 

Model hyperparameters were adopted as is from the the literature, and grid search was used to tune optimizer hyperparameters.

For full details on each and every experiment run, please refer to [`docs/Data Mining Final Report.pdf`](docs/Data%20Mining%20Final%20Report.pdf).

### Repository Structure

The `src/` folder contains all of the code, and is organized in the following structure:

1. `src/dataloaders.py`
    
    Contains dataloaders which parse the two datasets (100k, 1M) and generate train and test sets. The regular classes return `(userid, itemid, rating)` triplets (for RMSE and BCE loss functions), whereas the pairwise classes return `userid, positive_itemid, negative_itemid` triplets (for BPR loss).

2. `src/metrics.py`

    Contains implementations of the Hit Ratio and NDCG metrics.

3. `src/models.py`

    Contains implementations of models. The clases are named after the combination of the modeling algorithm and the loss function used. The following models have been implemented:

    - Matrix Factorization with RMSE loss (MFRMSE)
    - Matrix Factorization with BPR loss (MFBPR)
    - Neural Matrix Factorization with BCE loss (NMFBCE)
        - Combines the Generalized Matrix Factorization (GMF) model with the Multilayer Perceptron Model (MLP), which may or may not be pretrained.

4. `src/trainer.py`

    Contains a single class that implements the training and evaluation loop.

The notebooks in the root folder contain experiments that use the above modules, and can be identified by the same nomenclature as that of the models.

`gridsearch_logs/` contains evaluation data generated during hyperparameter tuning.

`saved_models/` contains pretrained models that were saved based on the optimal hyperparameters found via grid search.
