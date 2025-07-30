######### Protein Binding Site Predictor #########

Overview:

This program predicts protein binding sites using a pre-trained machine learning model. It takes as input a Protein Data Bank (PDB) file or folder containing Protein Data Bank (PDB) folder and outputs three files per PDB file
	1) CSV file containing the predictions for both binding (1) and non binding residuals.
	2) CSV file with a list of the binding residuals.
	3) PDB file containing the predicted binding site residues.

Requirements:

    Python 3.x
    mkdssp version 3.0.0
    Required Python packages:
        scikit-learn (sklearn)
    	NumPy (numpy)
    	pandas (pandas)
        Biopython (biopython)
        joblib (joblib)

Usage:

To use the program, follow these steps:

    # 1) Install the required Python packages using pip:

		pip install biopython scikit-learn pandas joblib

    # 2) Run the program with the following command-line arguments:

		python main.py -model <path_to_model_file> -pdb <path_to_pdb_file_or_folder> -out <output_folder_path>
		
    If no folder is provided, it will be created

    -model: Path to the trained machine learning model file.
    -pdb: Path to the PDB file of the protein structure or folder with PDB files.
    -out: Path to the output folder where predictions will be written.

Example:

    python main.py -model model.pkl -pdb protein_structure.pdb -out predictions/

    The program will generate three output files:
        <output_file_path>.csv: CSV file containing the predictions.
        <output_file_path>_binding.csv: CSV file containing a list of the binding site residuals.
        <output_file_path>_binding.pdb: PDB file containing the predicted binding site residues.

Additional Notes:

    Ensure that the current folder has the file atom_types.csv.
    Ensure that the input PDB file contains valid protein structure data.
    The trained model file must be provided for prediction.
    
    
    
######## Model Generator #########

Overview:

This script generates a Random Forest machine learning model for predicting protein binding sites using features extracted from PDB (Protein Data Bank) files. It outputs:
	1) Random Forest model file with the specified name.
	2) If the -test flag is provided; the test dataset is saved to evaluate the model post-hoc.
    
Requirements:

    Python 3.x
    mkdssp version 3.0.0
    Required Python packages:
    	scikit-learn (sklearn)
    	NumPy (numpy)
    	pandas (pandas)
    Required Modules:
        extract_features.py module
        random_forest.py module

Usage:

	python3 model_generator.py -pdb <pdb_dir> -labels <labels_dir> -out <output_file> -test <test_file>

    -pdb: Directory containing the PDB files.
    -labels: Directory containing the CSV files with classification labels.
    -out: File path to save the machine learning model.
    -test (optional): File path to save the test dataset to test accuracy.

Example:

    python3 model_generator.py -pdb pdb_data/ -labels labels/ -out model.pkl -test test_data.csv

Additional Notes
    Ensure that the current folder has the file atom_types.csv.

######### Accuracy Evaluation #########

This script evaluates a Random Forest model on a test dataset and optionally displays various quality measures, such as accuracy, AUC, precision, recall, F1-score, and confusion matrix. Additionally, it can plot the ROC curve and save it as an image file. Output:

    1) If the -metrics flag is set, the script prints the following quality measures:
        Accuracy
        AUC (Area Under the Curve)
        Precision
        Recall
        F1-score
        Confusion matrix (displayed as a heatmap)

    2) If the -roc flag is provided, the ROC curve is plotted and saved as the specified image file.

    3) If the -pred flag is provided, the predictions are saved to the specified CSV file.


Requirements:

    Python 3.x
    Required Python Packages:
    	scikit-learn (sklearn)
    	pandas 
    	matplotlib 
    	seaborn 

Usage:

	python evaluate_model.py -model -model <model_file> -test <test_set_file> -metrics -roc <roc_output_file> -pred <predictions_file>

    -model: File path to the trained machine learning model.
    -test: File path to the test dataset.
    -metrics: Flag to display accuracy, AUC, precision, recall, F1-score, and confusion matrix.
    -roc: File path to save the ROC curve plot.
    -pred: File path to save the predictions.

Example:

    python evaluate_model.py -model model.pkl -test test_set.csv -metrics -roc roc_curve.png -pred predictions.csv

## Authors

- Alex Ascunce
- María Chacon
- Paula Delgado
