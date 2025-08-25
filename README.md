# Protein Binding Site Predictor 
BSPredictor
## PREDICTOR:
This script predicts protein binding sites using a pre-trained Random Forest model. It takes as input a Protein Data Bank (PDB) file or folder containing PDB files and outputs three files per input file
1. CSV file containing the predictions for both binding (1) and non binding (0) residues.
2. CSV file with a list of the binding residues.
3. PDB file containing the predicted binding site residues.

### Requirements:

    Python 3.x
    mkdssp version 3.0.0
    Required Python packages:
        - scikit-learn (sklearn)
    	- NumPy (numpy)
    	- pandas (pandas)
        - Biopython (biopython)
        - joblib (joblib)

### Usage:

To use the program, follow these steps:
1) Clone this repository:
```bash
git clone https://github.com/mariachaort/SBI-PYT.Project.git
```
2) Install the required Python packages using pip:
```bash
pip install biopython scikit-learn pandas joblib
```
3) Run the program with the following command-line arguments:
```bash
python main.py -model <path_to_model_file> -pdb <path_to_pdb_file_or_folder> -out <output_folder_path>
```

If no folder is provided, it will be created.

Args:
1. -model: Path to the trained machine learning model file.
2. -pdb: Path to the PDB file of the protein structure or folder with PDB files.
3. -out: Path to the output folder where predictions will be written.

### Example:
```bash
python main.py -model model.pkl -pdb protein_structure.pdb -out predictions/
```

The program will generate three output files:
- <output_file_path>.csv: CSV file containing the predictions.
- <output_file_path>_binding.csv: CSV file containing a list of the binding site residuals.
- <output_file_path>_binding.pdb: PDB file containing the predicted binding site residues.

### Additional notes:
Ensure that the current folder has the file atom_types.csv.
Ensure that the input PDB file contains valid protein structure data.
The trained model file must be provided for prediction.
    
## MODEL GENERATOR
This script generates a Random Forest machine learning model for predicting protein binding sites using features extracted from PDB files. It outputs:
1.  Random Forest model file with the specified name.
2.  If the -test flag is provided; the test dataset is saved to evaluate the model post-hoc.
   
### Requirements
    Required Modules:
        extract_features.py module
        random_forest.py module

### Usage
```bash
python3 model_generator.py -pdb <pdb_dir> -labels <labels_dir> -out <output_file> -test <test_file>
```
Args:
1. -pdb: Directory containing the PDB files.
2. -labels: Directory containing the CSV files with classification labels.
3. -out: File path to save the machine learning model.
4. -test (optional): File path to save the test dataset to test accuracy.

### Example
```bash
python3 model_generator.py -pdb pdb_data/ -labels labels/ -out model.pkl -test test_data.csv
```
### Additional Notes
Ensure that the current folder has the file atom_types.csv.

## ACCURACY EVALUATION
This script evaluates a Random Forest model on a test dataset and optionally displays various quality measures, such as accuracy, AUC, precision, recall, F1-score, and confusion matrix. Additionally, it can plot the ROC curve and save it as an image file. Output:
1. If the -metrics flag is set, the script prints the following quality measures:
	- Accuracy
	- ROC_AUC (Area Under the ROC Curve)
	- Precision
	- Recall
	- F1-score
	- Confusion matrix
2. If the -roc flag is provided, the ROC curve is plotted and saved as the specified image file.
3.If the -pred flag is provided, the predictions are saved to the specified CSV file.

### Requirements:
    Python 3.x
    Required Python Packages:
    	scikit-learn (sklearn)
    	pandas 
    	matplotlib 
    	seaborn 

### Usage:
```bash
python evaluate_model.py -model -model <model_file> -test <test_set_file> -metrics -roc <roc_output_file> -pred <predictions_file>
```
Args:
1. -model: File path to the trained machine learning model.
2. -test: File path to the test dataset.
3. -metrics: Flag to display accuracy, AUC, precision, recall, F1-score, and confusion matrix.
4. -roc: File path to save the ROC curve plot.
5. -pred: File path to save the predictions.

### Example:
```bash
python evaluate_model.py -model model.pkl -test test_set.csv -metrics -roc roc_curve.png -pred predictions.csv
```

## Authors

Equally contributed to the project:
- Alex Ascunce
- María Chacón
- Paula Delgado
