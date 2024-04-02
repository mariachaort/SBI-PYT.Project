################################################
####        MAIN PROGRAM - PREDICTOR        ####
################################################

# Predict protein binding sites using a pre-trained model.

# Options:
  # -h, --help   show this help message and exit
  # -model       Path to the trained model file.
  # -pdb         Path to the PDB file or folder with pdb files.
  # -out         Path to the output folder where predictions will be written.

import argparse
import os
import sys
import pandas as pd
import joblib
import warnings
from extract_features import *
from Bio.PDB import PDBParser
from Bio import BiopythonWarning

warnings.simplefilter('ignore', BiopythonWarning)

def predict_binding_sites(model_file, pdb_input, output_folder, data_dir="."):
    try:
        # Verify if a model is provided
        if not os.path.isfile(model_file):
            raise FileNotFoundError("A valid model file is needed for prediction")

        # Load the trained model
        model = joblib.load(model_file)

        # Check if pdb_input is a single file or a folder
        if os.path.isfile(pdb_input):
            pdb_files = [pdb_input]
        elif os.path.isdir(pdb_input):
            pdb_files = [os.path.join(pdb_input, f) for f in os.listdir(pdb_input) if f.endswith('.pdb')]
        else:
            raise ValueError("Invalid input provided. Please provide a valid PDB file or folder.")

        # Iterate over each PDB file
        for pdb_file in pdb_files:
            # Extract protein ID from the filename
            protein_id = os.path.splitext(os.path.basename(pdb_file))[0]
            sys.stderr.write("###Processing protein: " + protein_id + "\n")

            # Extract features from the PDB file
            sys.stderr.write("###Extracting features from " + str(pdb_file) + "\n")
            features1 = ExtractPDBFeatures(pdb_file)
            features2 = extract_dssp_features(pdb_file)
            compute_prop_df = compute_properties(data_dir, pdb_file)
            neighbors_results = find_neighbors(pdb_file)
            features3 = weighted_average(compute_prop_df, neighbors_results)
            sys.stderr.write("###Features correctly extracted\n")

            # Combine all features into a single DataFrame
            all_features = merge_dataframes([features1, features2, features3], 'res_name')
  
            # Encode categorical variables as 0 and 1
            df_encoded = pd.get_dummies(all_features, columns=['Residue name', 'SS'])
            boolean_columns = df_encoded.select_dtypes(include=bool).columns
            df_encoded[boolean_columns] = df_encoded[boolean_columns].astype(int)
        
            # Order columns
            model_order = ['res_name','Cter_dist', 'Nter_dist', 'ASA', 'Phi Angle', 'Psi Angle',
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
        
            # Make predictions
            predictions = model.predict(df_encoded)
            probabilities = model.predict_proba(df_encoded)[:, 1]  # Probability of the positive class
            sys.stderr.write("###Predictions correctly made\n")

            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
                
            # Write predictions for each residue (to file)
            with open(os.path.join(output_folder, protein_id + '_predictions.csv'), 'w') as f_out:
                f_out.write("res_name,binding_site,probability\n")
                for i, (prediction, probability) in enumerate(zip(predictions, probabilities)):
                    residue_name = df_encoded.index[i]
                    f_out.write(f"{residue_name},{prediction},{probability}\n")
                    
            # List to store binding site residues
            binding_site_residues = []

            # Append predicted binding site residues to the list
            for i, (prediction, probability) in enumerate(zip(predictions, probabilities)):
                residue_name = df_encoded.index[i]
                if prediction == 1 and probability > 0.80:  # Filter by probability
                    binding_site_residues.append(residue_name)

            # Write PDB file with only binding site residues if there are any
            if binding_site_residues:
                write_predictions(binding_site_residues, pdb_file, os.path.join(output_folder, protein_id + '_binding_sites.pdb'))

                # Write CSV file for binding sites if there are any
                with open(os.path.join(output_folder, protein_id + '_binding_sites.csv'), 'w') as f_binding_out:
                    f_binding_out.write("res_name\n")
                    for residue_name in binding_site_residues:
                        f_binding_out.write(f"{residue_name}\n")
            else:
                sys.stderr.write("###No binding site residues predicted with probability higher than 0.8\n")

        sys.stderr.write("###Output files saved\n")

    except FileNotFoundError as e:
        print("Error:", e)
    except ValueError as e:
        print("Error:", e)
    except Exception as e:
        print("An error occurred:", e)
        

def write_predictions(binding_site_residues, pdb_file, output_file):
    """
    Create a file containing the predicted binding site residues in PDB format.

    Args:
    - binding_site_residues (list): A list of residue names considered as binding site residues.
    - pdb_file (str): The path to the PDB file.
    - output_file (str): The path to the output file where the predictions will be written.

    Output:
    - A file containing the predicted binding site residues in PDB format.
    """

    try:
        # Parse the PDB file to obtain its structure
        parser = PDBParser()
        basename = os.path.basename(pdb_file)
        pdb_id = os.path.splitext(basename)[0]
        structure = parser.get_structure(pdb_id, pdb_file)
        
        # Open the PDB file for reading and the output file for writing
        with open(pdb_file, "r") as PDB:
            with open(output_file, "w") as output:
                # Iterate through each line in the PDB file
                for line in PDB:
                    # Check if the line represents an ATOM entry
                    if line.startswith("ATOM"):
                        # Extract relevant information from the line
                        name = line[17:20].strip()
                        num = line[23:26].strip()
                        chain = line[21].strip()
                        residue_name = pdb_id + '_' + chain + '_' + name + '_' + num
                        # Write the line to the output file if the residue is predicted as a binding site
                        if residue_name in binding_site_residues:
                            output.write(line)
        

    except FileNotFoundError as e:
        print("Error: PDB file not found:", e)
    except Exception as e:
        print("An error occurred:", e)

if __name__ == "__main__":
    # Define command-line arguments
    parser = argparse.ArgumentParser(description="Predict protein binding sites using a pre-trained model.")
    parser.add_argument("-model", dest="model_file", help="Path to the trained model file.")
    parser.add_argument("-pdb", dest="pdb_input", help="Path to the input PDB file or folder containing PDB files.")
    parser.add_argument("-out",dest="output_folder", help="Directory path to save the predictions.")
    args = parser.parse_args()    
    
    # Call the predict_binding_sites function with the provided arguments
    predict_binding_sites(args.model_file, args.pdb_input, args.output_folder)
