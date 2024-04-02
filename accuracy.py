################################
####        ACCURACY        ####
################################

# Evaluate Random Forest model on test set and optionally display quality measures and plot ROC curve.

# Options:
  #-h, --help         show this help message and exit
  #-model MODEL_FILE  File path to the trained machine learning model.
  #-test TEST_FILE    File path to the test dataset.
  #-metrics           Flag to display accuracy, AUC, precision, recall, F1-score, and confusion matrix.
  #-roc ROC_OUTPUT    File path to save the ROC curve plot.
  #-pred PREDICTIONS  File path to save the predictions.

import argparse
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score, roc_curve, auc, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def main(model_file, test_file, metrics_flag, roc_output, predictions):
    """
    Main function to evaluate the Random Forest model on the test set and optionally display accuracy, AUC, 
    save ROC curve, and generate a file with predictions.

    Args:
        model_file (str): File path to the trained machine learning model.
        test_file (str): File path to the test dataset.
        metrics_flag (bool): Flag to indicate whether to display accuracy, AUC, precision, recall, F1-score,
                             and confusion matrix.
        roc_output (str): File path to save the ROC curve plot.
        predictions (str): File path to save the predictions.
    """
    # Load the model from file
    with open(model_file, 'rb') as file:
        rf_fit = pickle.load(file)

    # Load the test dataset
    test_df = pd.read_csv(test_file)
    # Set 'res_name' as index
    test_df.set_index('res_name', inplace=True)
    # Separate features (X) and target variable (y) from the test set
    X_test = test_df.drop(columns=['class'])
    y_test = test_df['class']

    # Make predictions on the test set
    y_prob = rf_fit.predict_proba(X_test)
    
    # Filtrar las predicciones por probabilidad
    threshold = 0.7  # Umbral de probabilidad deseado
    y_pred_filtered = (y_prob[:, 1] > threshold).astype(int)
    
    # Calculate Area Under the Curve (AUC)
    fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])
    roc_auc = auc(fpr, tpr)
	
    if roc_output:
        # Plot the ROC curve and save it
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.savefig(roc_output)

    if predictions:
        # Generate a DataFrame with predictions
        predictions_df = pd.DataFrame({'class': y_test, 'predicted': y_pred_filtered})
        # Save the predictions to a CSV file with 'res_name' as index
        predictions_df.to_csv(predictions)
    
    if metrics_flag:
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred_filtered)
        print("Accuracy:", accuracy)

        # Print AUC
        print("AUC:", roc_auc)

        # Calculate precision, recall, and F1-score
        precision = precision_score(y_test, y_pred_filtered)
        print("Precision:", precision)
        
        recall = recall_score(y_test, y_pred_filtered)
        print("Recall:", recall)
        
        f1 = f1_score(y_test, y_pred_filtered)
        print("F1-score:", f1)
        
        # Generate and print confusion matrix
        cm = confusion_matrix(y_test, y_pred_filtered)
        cm = confusion_matrix(y_test, y_pred_filtered)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.show()
        plt.show(block=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Random Forest model on test set and optionally display accuracy, AUC, save ROC curve, and generate a file with predictions.")
    parser.add_argument("-model", dest="model_file", help="File path to the trained machine learning model.")
    parser.add_argument("-test", dest="test_file", help="File path to the test dataset.")
    parser.add_argument("-metrics", action='store_true', help="Flag to display accuracy, AUC, precision, recall, F1-score, and confusion matrix.")
    parser.add_argument("-roc", dest="roc_output", help="File path to save the ROC curve plot.")
    parser.add_argument("-pred", dest="predictions", help="File path to save the predictions.")
    args = parser.parse_args()

    main(args.model_file, args.test_file, args.metrics, args.roc_output, args.predictions)



