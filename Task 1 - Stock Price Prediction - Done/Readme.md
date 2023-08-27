# Reliance Stock Price Prediction
This notebook builds a model to predict the future stock price of Reliance using Long Short-Term Memory (LSTM) neural networks.

## Data
The historical daily stock price data for Reliance (RELIANCE.NS) is downloaded from Yahoo Finance for the period from December 18, 2005 to August 11, 2023. The data includes the following fields:

Date
Open
High
Low
Close
Adjusted Close
Volume
The data is loaded and preprocessed to extract just the Date and Close price, which are used for training the model.

## Model
The LSTM model is defined as follows:

LSTM layer with 128 units and return sequences set to True
Dropout layer with 0.2 dropout
LSTM layer with 128 units
Dropout layer with 0.2 dropout
LSTM layer with 64 units
Dropout layer with 0.2 dropout
Output dense layer with 1 unit
The model is trained for 50 epochs and RMSE on the test set is calculated.

## Results
The model achieves a test RMSE of around 65.6 on predicting close prices. The predictions on both test data and future data are plotted to visualize how well the model fits the actual data.

The model is also saved so it can be loaded later for predictions.

Usage
To use the trained model for predictions:

Load the model:
<!---->
Copy code

```python
  from tensorflow.keras.models import load_model
  model = load_model('Models/Final_fast.pb')
```


2.  Prepare input data in the same preprocessed format as used during training (sequence of 60 prior days' close prices, scaled to 0-1)
3.  Call model.predict() and invert the scaling

The notebook also shows how to generate multiple future predictions by feeding predictions back into the model.

Next Steps
Some ways to improve the model:

Try different model architectures (more layers, different layer sizes etc)
Tune hyperparameters like batch size, dropout
Use regularization techniques like early stopping
Use other data like technical indicators along with closing price
Ensemble models for more robust predictions
