Certainly! Here's the combined Markdown file with the content from the three provided files:

---

# Reliance Stock Price Prediction

This notebook builds a model to predict the future stock price of Reliance using Long Short-Term Memory (LSTM) neural networks.

## Data

The historical daily stock price data for Reliance (RELIANCE.NS) is downloaded from Yahoo Finance for the period from December 18, 2005 to August 11, 2023. The data includes the following fields:

- Date
- Open
- High
- Low
- Close
- Adjusted Close
- Volume

The data is loaded and preprocessed to extract just the Date and Close price, which are used for training the model.

## Model

The LSTM model is defined as follows:

- LSTM layer with 128 units and return sequences set to True
- Dropout layer with 0.2 dropout
- LSTM layer with 128 units
- Dropout layer with 0.2 dropout
- LSTM layer with 64 units
- Dropout layer with 0.2 dropout
- Output dense layer with 1 unit

The model is trained for 50 epochs and RMSE on the test set is calculated.

## Results

The model achieves a test RMSE of around 65.6 on predicting close prices. The predictions on both test data and future data are plotted to visualize how well the model fits the actual data.

The model is also saved so it can be loaded later for predictions.

**Usage**

To use the trained model for predictions:

Load the model:

```python
from tensorflow.keras.models import load_model
model = load_model('Models/Final_fast.pb')
```

Prepare input data in the same preprocessed format as used during training (sequence of 60 prior days' close prices, scaled to 0-1).

Call `model.predict()` and invert the scaling.

The notebook also shows how to generate multiple future predictions by feeding predictions back into the model.

**Next Steps**

Some ways to improve the model:

- Try different model architectures (more layers, different layer sizes, etc.).
- Tune hyperparameters like batch size, dropout.
- Use regularization techniques like early stopping.
- Use other data like technical indicators along with closing price.
- Ensemble models for more robust predictions.

# Titanic Survival Prediction Model

This project builds a binary classification model to predict whether a passenger survived the Titanic disaster based on attributes like passenger class, age, gender etc.

## Data

The dataset used is the classic Titanic dataset from Kaggle. It contains demographics and passenger information from 891 of the 2224 passengers and crew on board the Titanic.

The data dictionary is:

- **Survived**: Outcome of survival (0 = No, 1 = Yes)
- **Pclass**: Socio-economic class (1 = Upper, 2 = Middle, 3 = Lower)
- **Name**: Name of passenger
- **Sex**: Sex of the passenger
- **Age**: Age of the passenger (Some entries contain `NaN`)
- **SibSp**: Number of siblings and spouses of the passenger aboard
- **Parch**: Number of parents and children of the passenger aboard
- **Ticket**: Ticket number of the passenger
- **Fare**: Passenger fare (Some entries contain `NaN`)
- **Cabin**: Cabin number of the passenger (Some entries contain `NaN`)
- **Embarked**: Port of embarkation of the passenger (C = Cherbourg, Q = Queenstown, S = Southampton)

## Model

The preprocessed data is used to train a deep neural network model with the following architecture:

- Input layer
- Dense layer (128 units, ReLU activation)
- Batch Normalization
- Dropout (0.3)
- Dense layer (64 units, ReLU activation)
- Batch Normalization
- Dropout (0.3)
- Output layer (1 unit, Sigmoid activation)

The model is trained for 500 epochs with a batch size of 32.

## Results

The model achieves an accuracy of ~82% on the test set.

The trained model is saved to disk and can be loaded for inference on new data.

**Usage**

To load the trained model and make predictions:

```python
import tensorflow as tf

model = tf.keras.models.load_model("Models/titan_titanic_train_500_32_8_relu_relu_1692464561")

# Prepare input data and make predictions
predictions = model.predict(input_data)
```

**Next Steps**

Some ways to improve the model:

- Try different model architectures like CNN, RNN etc.
- Tune hyperparameters like batch size, epochs, dropout rate etc.
- Use regularization techniques like early stopping to prevent overfitting.
- Experiment with balancing the training data distribution.
- Use ensembling techniques to combine multiple models.

# Handwriting Classification using Convolutional Neural Networks

This project demonstrates the process of classifying handwritten digits from the MNIST dataset using Convolutional Neural Networks (CNNs) with TensorFlow and Keras.

## Introduction

This project focuses on the classification of handwritten digits from the MNIST dataset, which is a widely used benchmark in the field of machine learning. The goal is to build a Convolutional Neural Network (CNN) model that can accurately predict the digits based on their images.

## Prerequisites

- Python (3.x recommended)
- TensorFlow (2.x)
- Pandas
- Matplotlib
- Numpy
- Scikit-learn (for t-SNE visualization)

You can install the required libraries using the following command:

```bash
pip install tensorflow pandas matplotlib numpy scikit-learn
```

## Installation

1. Clone or download this repository to your local machine.
2. Navigate to the project directory:

   ```bash
   cd Handwriting-Classification-CNN
   ```

## Usage

1. Run the Jupyter Notebook or Python script in your preferred development environment.
2. The code consists of the following main sections:
   - Loading and Preprocessing the MNIST dataset
   - Visualizing Sample Images
   - Visualizing using t-SNE
   - Building and Training the CNN Model

## Results

The trained CNN model should achieve a high accuracy on the MNIST test dataset. The t-SNE visualization provides insights into the distribution of data points in a lower-dimensional space, highlighting the clusters formed by different digits.

## Contributing

Contributions to this project are welcome! If you have any improvements or bug fixes, feel free to submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).

---
