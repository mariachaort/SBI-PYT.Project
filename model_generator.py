#######################################
####        MODEL GENERATOR        ####
#######################################

# Generate Random Forest model for protein binding prediction.

# Options:
  # -h, --help          show this help message and exit
  # -pdb PDB_DIR        Directory containing the PDB files.
  # -labels LABELS_DIR  Directory containing the CSV files with classification labels.
  # -out OUTPUT_FILE    File path to save the machine learning model.
  # -test TEST_FILE     File path to save the test dataset (optional)

# Its necessary to have the file atom_types.csv in the current folder.

import argparse
import os
import sys
import pandas as pd
from extract_features import *
from sklearn.model_selection import train_test_split
from random_forest import RandomForestModelCreator

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

def main(pdb_dir, labels_dir, output_file, test_file):
    """
    Main function to generate Random Forest model for protein binding prediction.

    Args:
        pdb_dir (str): Directory containing the PDB files.
        labels_dir (str): Directory containing the CSV files with classification labels.
        output_file (str): File path to save the machine learning model.
        test_file (str): File path to save the test dataset.
    """
    # Obtain dataframe with all protein features from all proteins of the database
    df = extract_features(pdb_dir, labels_dir)
    if not df.empty:
        # Take into account unbalanced binding vs non-binding residuals -> Balance dataset
        class_counts = df['class'].value_counts()
        min_count = min(class_counts)
        df_filtered = df[df['class'].isin([0, 1])]
        df_balanced = df_filtered.groupby('class').apply(lambda x: x.sample(min_count))
        df_balanced.reset_index(drop=True, inplace=True)
	
        # Encode categorical variables as 0 and 1
        df_encoded = pd.get_dummies(df_balanced, columns=['Residue name', 'SS'])
        boolean_columns = df_encoded.select_dtypes(include=bool).columns
        df_encoded[boolean_columns] = df_encoded[boolean_columns].astype(int)
        
        # Order columns
        model_order = ['class','res_name','Cter_dist', 'Nter_dist', 'ASA', 'Phi Angle', 'Psi Angle',
                    'NH->O_1_relidx', 'NH->O_1_energy', 'O->NH_1_relidx', 'O->NH_1_energy',
                    'NH->O_2_relidx', 'NH->O_2_energy', 'DON', 'HPB', 'ACP', 'NEG', 'ARM',
                    'POS', 'Residue name_ALA', 'Residue name_ARG', 'Residue name_ASN',
                    'Residue name_ASP', 'Residue name_CYS', 'Residue name_GLN',
                    'Residue name_GLU', 'Residue name_GLY', 'Residue name_HIS',
                    'Residue name_ILE', 'Residue name_LEU', 'Residue name_LYS',
                    'Residue name_MET', 'Residue name_PHE', 'Residue name_PRO',
                    'Residue name_SER', 'Residue name_THR', 'Residue name_TRP',
                    'Residue name_TYR', 'Residue name_VAL', 'SS_-', 'SS_B', 'SS_E', 'SS_G',
                    'SS_H', 'SS_I', 'SS_S', 'SS_T']
        
        # If not present, add it with 0
        for col in model_order:
            if col not in df_encoded.columns:
                df_encoded[col] = 0

        df_encoded = df_encoded.reindex(columns=model_order)
        # Add the index
        df_encoded.set_index('res_name', inplace=True)

        # Separate train and test datasets
        train_df, test_df = train_test_split(df_encoded, test_size=0.3, random_state=42)

        # Generate random forest model
        RandomForestModelCreator(train_df, output_file)

        # Save test dataset as CSV
        test_df.to_csv(test_file, index=True)
        sys.stderr.write("###Test dataset saved as "+ str(test_file) +"\n")
        
    else:
        print("Error: DataFrame 'df' is empty. No features extracted.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Random Forest model for protein binding prediction.")
    parser.add_argument("-pdb", dest="pdb_dir", help="Directory containing the PDB files.")
    parser.add_argument("-labels", dest="labels_dir", help="Directory containing the CSV files with classification labels.")
    parser.add_argument("-out", dest="output_file", help="File path to save the machine learning model.")
    parser.add_argument("-test", dest="test_file", help="File path to save the test dataset.", required=False)
    args = parser.parse_args()

    main(args.pdb_dir, args.labels_dir, args.output_file, args.test_file)





