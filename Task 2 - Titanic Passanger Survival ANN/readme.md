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

## Usage

To load the trained model and make predictions:

```python
import tensorflow as tf

model = tf.keras.models.load_model("Models/titan_titanic_train_500_32_8_relu_relu_1692464561")

# Prepare input data and make predictions
predictions = model.predict(input_data)  
```

## Next Steps

Some ways to improve the model:

- Try different model architectures like CNN, RNN etc.
- Tune hyperparameters like batch size, epochs, dropout rate etc. 
- Use regularization techniques like early stopping to prevent overfitting.
- Experiment with balancing the training data distribution.
- Use ensembling techniques to combine multiple models.
