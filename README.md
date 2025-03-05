# Supreme Court Decision Prediction Using Machine Learning

## Project Overview

This project aims to predict the outcomes of U.S. Supreme Court cases (whether the court will reverse or affirm a lower court's decision) using various machine learning models, including Logistic Regression (LR) and Deep Neural Networks (DNN). The dataset includes information about cases, the participants, and the justices involved. The project explores the performance of different models, with a particular focus on the effects of incorporating decision-related and political orientation features.

## Project Structure

- **Data/**: Contains the processed and raw datasets used in the project.

  - `dev_processed.csv`, `test_processed.csv`, `train_processed.csv`: Processed CSV files for the training, validation, and test sets.
  - `dev.jsonl`, `test.jsonl`, `train.jsonl`: JSONL files containing the raw court case data.
  - `sembed/`: Folder containing precomputed sentence embeddings for court discussions.

- **figures/**: Stores the figures generated during the experiments.

- **models/**: Stores trained model weights.

  - `best_kfold_model.pth`: The best performing Deep Neural Network model saved during k-fold cross-validation.
  - `model-1.pth` to `model-6.pth`: Checkpoints for different iterations of the DNN model.

- **notebooks/**: Contains Jupyter notebooks for the different experiments.

  - `A2-data-preprocessing.ipynb`: Notebook detailing the data preprocessing steps.
  - `A2-model-Baseline.ipynb`: Baseline models (Majority Class and Random Guessing) implementation.
  - `A2-model-DNN.ipynb`: Implementation of the Deep Neural Network model.
  - `A2-model-LR.ipynb`: Implementation of the Logistic Regression model.
  - `A2-overview.ipynb`: Provides a high-level overview of the project.

- **predictions/**: Contains CSV files with model predictions.

## How to Run the Project

1. **Data Preparation**:

   - The data has been preprocessed and split into `train`, `dev`, and `test` sets. These can be found in the `Data/` folder.

2. **Model Training**:

   - Jupyter notebooks for each model (DNN and LR) are provided in the `notebooks/` folder. The models can be retrained using the `A2-model-DNN.ipynb` and `A2-model-LR.ipynb` notebooks.
   - Baseline models (Majority Class and Random Guessing) are implemented in `A2-model-Baseline.ipynb`.

3. **Evaluation**:

   - Results are visualized through ROC curves, learning curves, and confusion matrices, available in the `figures/` folder.
   - K-fold cross-validation was used for the DNN model, and the best model is saved as `best_kfold_model.pth`.

4. **Predictions**:
   - Model predictions are saved as CSV files in the `predictions/` folder.
   - `pred.csv`: The best Predictions made by the DNN and submitted to Kaggle.

## Key Experiments

- **RQ1: Baseline Comparison**:
  - Comparing the performance of the Majority Class and Random Guessing models with Logistic Regression and DNN models.
- **RQ2: Decision-Related Features**:

  - Incorporating decision-related features (`hearing_length_months`, `majority_ratio`) into the DNN model to improve performance.

- **RQ3: Political Orientation**:
  - Examining the effect of political orientation (`liberal_ratio`) of justices on Supreme Court decisions using the DNN model.

## Requirements

- Python 3.8
- PyTorch
- scikit-learn
- pandas
- matplotlib
