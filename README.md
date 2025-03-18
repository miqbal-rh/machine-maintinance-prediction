
# Model Training and Evaluation

## Overview

This repository contains code to train and evaluate a machine learning model using PyTorch for time-series prediction. The model is built using LSTM (Long Short-Term Memory) networks. The project uses a dataset for training, validation, and testing.

## Setup and Installation

To get started, first clone this repository and install the necessary dependencies.

### Prerequisites

You will need the following Python packages:

- `numpy`
- `pandas`
- `torch`
- `sklearn`
- `matplotlib`

You can install them using `pip`:

```bash
pip install numpy pandas torch scikit-learn matplotlib
```

### Dataset

Download or prepare your dataset in CSV format with the following columns:

- `time`: Time steps of the sequence.
- `unit`: The unit or ID of the data (e.g., different machines or sensors).
- `RUL`: Remaining useful life (target variable).
- Other feature columns (e.g., sensor readings).

Ensure that the dataset is split into three parts: training, validation, and testing, stored in `train.csv`, `validate.csv`, and `test.csv`, respectively.

## Data Generation (Optional)

If you don't have an existing dataset or want to generate synthetic data for training, validation, and testing, you can use the `generate_synthetic_data.py` script. This script allows you to generate a synthetic time-series dataset that mimics real-world machine maintenance scenarios.

### Running `generate_synthetic_data.py`

To generate synthetic data, simply run the script:

```bash
python generate_synthetic_data.py
```

The script will generate three CSV files: `train.csv`, `validate.csv`, and `test.csv`. You can adjust the number of data points, number of features, and other parameters by modifying the script as needed.

### Parameters in `generate_synthetic_data.py`

- **Number of Units**: Number of machines or sensors to simulate.
- **Number of Time Steps**: Number of time steps per unit (typically for a sequence).
- **Feature Generation**: Generates synthetic features that simulate sensor data (e.g., temperature, pressure).
- **RUL Generation**: Simulates the remaining useful life (`RUL`) of each unit, which is used as the target variable for model training.

Once the data is generated, it can be used for training, validation, and testing of the model.

## Data Preprocessing

The dataset needs preprocessing to ensure that there are no missing values, and the features are scaled appropriately. The following preprocessing steps are performed:

1. **Missing Value Handling**: Missing values are forward-filled.
2. **Scaling**: All feature columns are standardized using `StandardScaler` from `sklearn`.
3. **Feature Selection**: Only relevant features (excluding `time`, `unit`, and target `RUL`) are used for training.

The code reads the dataset, handles missing values, and scales the features before training.

## Model Architecture

The model is based on an LSTM architecture. Here's the structure:

- **LSTM Layer**: The core of the model for time-series data.
- **Fully Connected (FC) Layer**: Maps the LSTM output to the target value (`RUL`).
- **Activation**: No explicit activation function after the LSTM layer. The output is directly fed into the FC layer for regression.

### Hyperparameters

- **Hidden Size**: 64 units in the LSTM layer.
- **Number of Layers**: 2 layers for the LSTM.
- **Learning Rate**: 0.0001 for optimization.

## Model Training

The training loop includes:

- **Loss Function**: Mean Squared Error (MSE) loss, suitable for regression tasks.
- **Optimizer**: Adam optimizer with a learning rate of 0.0001.
- **Gradient Clipping**: To prevent exploding gradients during training.

### Training Procedure

The model is trained using the training dataset (`train.csv`) and evaluated on the validation dataset (`validate.csv`). The training procedure involves:

1. Loading the data.
2. Preprocessing the data (scaling and missing value handling).
3. Training the model on the training set.
4. Evaluating the model on the validation set.

```python
# Example usage for training the model:
train_model(train_loader, model, criterion, optimizer, device, num_epochs=10)
evaluate_model(val_loader, model, criterion, device)
```

## Model Saving

After the model has been trained, it is saved to disk to allow for later use or deployment. The model's state dictionary (weights and biases) is saved in a `.pth` file format. This can be used to load the model for inference or further training without retraining the entire model.

The saving process occurs at the end of the training procedure. By saving the model, you can easily deploy it to a production environment or continue training it at a later stage.
