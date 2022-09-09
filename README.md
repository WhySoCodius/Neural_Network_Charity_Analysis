# Neural_Network_Charity_Analysis

Python's TensorFlow platform is used to implement neural networks. Create a binary classifier that can determine if applicants will be approved for funding by the non-profit philanthropic foundation Alphabet Soup by using the attributes in the provided dataset.

## Overview

The goal of this project is to develop a binary classifier that can determine whether applications will be sponsored by a hypothetical company called "Alphabet Soup" or not. The preprocessing of the data is followed by the creation, training, and evaluation of a neural network model. Last but not least, an effort is made to improve the model by raising accuracy to 75% or greater.

## Resources
### Data Source

The [charity_data.csv](/Resources/charity_data.csv) was retrieved.

### Softwares

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)

## Results

### **Data Preprocessing**
"EIN" and "NAME" columns were dropped from the dataset after the [charity_data.csv](/Resources/charity_data.csv) file was read into a Pandas DataFrame. There was nothing of value in these identification columns.

The rare categorical variables in columns with more than 10 distinct values, such as the "CLASSIFICATION" and "APPLICATION TYPE" columns, have been binned into a new "other" column. OneHotEncoder is then used to encode a list of categorical variables. A brand-new dataframe is expanded with these encoded variable names. The original features are lost after combining the one-hot encoded features. 

"IS SUCCESSFUL" is the variable that this model considers to be its target; all other variables in the dataframe are thought of as features. The StandardScaler module from Scikit-Learn is used to standardise the numerical variables.

### **Compiling, Training, and Evaluating the Model**
A deep learning model is created to predict whether a "Alphabet Soup" sponsored organisation will be successful based on the features in the dataset after the data has been preprocessed. 

The activation function for the first and second hidden layers is Rectified Linear Unit (ReLU). A sigmoid activation function is employed for the output layer. Every five training epochs, a callback saves the model's weights. The model's accuracy and loss values are assessed after training. The model's output is shown in the image below.

![AlphabetSoupCharity](/Resources/AlphabetSoupCharity.png)

#### **Optimization Attempt 1**
The number of neurons in the hidden layers of this model were raised in the first optimization process. This had very little impact on the model's precision. For the other optimization efforts, the number of neurons is reset to their initial values. The total parameters became 10,086 with neurons as follows:

- hidden_nodes_layer1 = 110
- hidden_nodes_layer2 = 45
- hidden_nodes_layer3 = 10

![AlphabetSoupCharity_01](/Resources/AlphabetSoupCharity_01.png)

Since the accuracy is below 75% will attempt another level of optimisation.

#### **Optimization Attempt 2**
The SPECIAL_CONSIDERATIONS_N row was dropped as it was redundant. Activation function for hidden and input layers was changed to tanh. The total parameters became 9,976 with neurons as follows:

- hidden_nodes_layer1 = 110
- hidden_nodes_layer2 = 45
- hidden_nodes_layer3 = 10

![AlphabetSoupCharity_02](/Resources/AlphabetSoupCharity_02.png)

Since the accuracy is still below 75% will attempt another level of optimisation.

#### **Optimization Attempt 3**
Binning threshold was reduced to increase the input variables such as application_counts & name_counts and additional hidden layer was used. The total number of inputs increased to 140 & parameters became 34,216 with neurons as 

- hidden_nodes_layer1 = 150 neurons
- hidden_nodes_layer2 = 75 neurons
- hidden_nodes_layer3 = 20 neurons
- hidden_nodes_layer4 = 10 neurons

![AlphabetSoupCharity_03](/Resources/AlphabetSoupCharity_03.png)

The target model performance was acheived and the final performance was 76.90 %


## Summary
In this investigation, optimization attempt succeeded in creating a model with a prediction accuracy level of 75% or above in the 3rd event against the initial 72.52%. The following techniques were tried: 

| Optimization Status | Accuracy |
| --- | --- |
| Before optimization     |72.52    |
| First Attempt	          |72.59    |	
| Second Attempt          |72.64    |	
| Third Attempt           |76.90    |	


- Adding additional hidden layers. 
- Additional neurons are added to hidden layers. 
- Making use of various activation mechanisms. 

There may be a variety of explanations for why this model was unable to achieve the desired prediction accuracy level of 75%. It might be necessary to modify the amount of neurons in the buried layers. During training, the number of epochs may need to be raised. There could be anomalies or factors in the input data that throw the model off.

In this binary classification job, a Random Forest Classifier may be preferred than a Deep Learning model. Random Forest Classifiers are far easier to set up and can be trained in a matter of seconds, in contrast to Deep Learning models, which require extensive setup and lengthy training. Furthermore, these two models' levels of prediction accuracy would probably be comparable. It is advised that any further work on this project make use of a Random Forest Classifier due to its simplicity of usage and suitability for this case.