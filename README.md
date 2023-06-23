# Fetal Health Classification:

This repository contains machine learning model for classifying fetal health based on the collected dataset from Kaggle ( [Dataset](https://www.kaggle.com/datasets/andrewmvd/fetal-health-classification) ). The models were trained and evaluated using various classification algorithms, including K-Nearest Neighbors (KNN), Support Vector Machines (SVM), Decision Tree, Random Forest, and Gradient Boosting classifiers. The best-performing model was determined to be the Gradient Boosting Classifier( n_estimators=400,learning_rate=0.1 ).

## Repository Contents

- `data/`: This directory contains the dataset files obtained from Kaggle.
- `main.py`: This python file contains all the code realted to the pre-processing, evaluating, hyperparameter tuning and training.
- `fetal_health_model.pkl`: Trained model .pkl file.
- `README.md`: This file, providing an overview of the repository and its contents.

## Required Python Libraries

- sklearn
- numpy
- pandas
- pickle
- matplotlib

Make sure these libraries are installed in your Python environment before running the notebooks.

## Implementation:

### Train Test Splitting:

First we split the data into training and testing data with ration of 4:1 and stratified.

### Scaling:

The the data is scaled using Standardization, for this I have used SK learn StandardScaler.

### Model Evaluation:

The best performing model is selected using KFold(split=5) cross validation using the following models.

- KNN
- SVM
- Decision Tree Classifier
- Random Forest Classifier
- Gradient Boosting Classifier
<div align="center">
    <img src="https://github.com/KishorBalgi/ML-Fetal-Health-Classification/assets/75678927/07be37f5-65e7-4635-ad90-625aa897c43b" alt="Image description">
</div>

From the scores optained and plotting a box plot of these scores we can observe that Gradient Boosting Classifier performs better than other classifiers.

### Hyperparameter Tuning:

Now, we have selected Gradient Boosting Classifier(GBC) to train our model, it is necessary to choose the right hyperparameters, this is done with the help of GridSearchCV and RandomizedSearchCV.

<div align="center">
    <img src="https://github.com/KishorBalgi/ML-Fetal-Health-Classification/assets/75678927/f452941e-10cf-4a0a-aa9a-6be9b159ee72" alt="Image description">
</div>

From the scores obtained from the GridSearchCV and RandomizedSearchCV the GCV gives better results as compared to RCV.

We observe that the Learing Rate = 0.1 and n-estimators = 400 gives the best score of .9488 .

### Train the model:

we can now train the model using the best parameters obtained form hyperparameter tuning ( i.e, Learing Rate = 0.1 and n-estimators = 400)

Accuracy: 0.95

### Score:

<div align="center">
    <img src="https://github.com/KishorBalgi/ML-Fetal-Health-Classification/assets/75678927/4b28beab-1507-464e-890c-5054022b1ccf" alt="Image description">
</div>

## Usage

To utilize the models and reproduce the results:

1. Download the dataset from Kaggle and place it in the `data/` directory.
2. In the main.py file some line are commentd which are simply used to either pre-processing, standardization, evaluation, hyperparameter tuning. Uncomment these line as per requirement.

## License

The code in this repository is licensed under the [MIT License](LICENSE). Feel free to use and modify the code for your purposes.
