####################################
####        RANDOM FOREST       ####
####################################
import sys
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV

#Function to generate the model and do a cross validation for hyperparameter tunning:
def RandomForestModelCreator(dataframe, output_file):
    '''
    Performs Random Forest machine learning algorithm to create and save the model and parameters.

    Args:
        dataframe (DataFrame): DataFrame containing the features and target variable.
        output_file (str): File path to save the machine learning model.

    Returns:
        Random Forest Model
    '''
    sys.stderr.write("###Random Forest Model is being generated... please be patient\n")
    # Random Forest model
    rf_model = RandomForestClassifier(n_estimators=100, criterion='gini', random_state=33)

    # Define recipe and pre-processing
    rf_recipe = Pipeline([
        ('impute', KNNImputer(n_neighbors=5))
    ])

    # Workflow
    rf_workflow = Pipeline([
        ('recipe', rf_recipe),
        ('rf', rf_model)
    ])

    # Tune the max_features parameter
    rf_grid = {'rf__max_features': np.arange(1, 38)}
    rf_tune_results = RandomizedSearchCV(
        rf_workflow, 
        param_distributions=rf_grid, 
        cv=5,
        scoring='roc_auc', 
        n_iter=37, 
        random_state=33
    )
    rf_tune_results.fit(dataframe.drop(columns=['class']), dataframe['class'])  # Utilize the training set for training

    # Select best parameters
    param_final = rf_tune_results.best_params_

    # Finalize workflow with best parameters
    rf_workflow.set_params(**param_final)

    # Fit the model
    rf_fit = rf_workflow.fit(dataframe.drop(columns=['class']), dataframe['class'])  # Utilize the training set for training

    # Inform user about model saving
    sys.stderr.write("###Random Forest machine learning model saved as: " + str(output_file) +  "\n")

    # Save the model to disk
    with open(str(output_file), 'wb') as file:
        pickle.dump(rf_fit, file)
