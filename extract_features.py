########################################
####        EXTRACT FEATURES        ####
########################################
	
	# 1) read_types (data_dir): reads a csv file located in data_dir and returns a dictionary with properties associated to each residue atom.
	# 2) get_residue_index (residue): generates a unique identifier for each residue in a PDB file.
	# 3) compute_properties (data_dir, pdb_file): computes properties of each residue in a PDB file.
	# 4) find_neighbors (pdb_file, radius=6.0): finds spatial neighbors for each atom in a PDB file within a 6A radius.
	# 5) weighted_average (df, neighbours_result): computes de weighted properties of residuals using the spatial neighbors.
	# 6) ExtrractPDBFeatures (pdb_file): extracts the sequence features of a PDB file.
	# 7) extract_dssp_features (pdb_file): extracts dssp features of a PDB file.
	# 8) merge_dataframes(df_list, on_column): merges dataframes using on_column.
	# 9) extract_features(pdb_dir, labels_dir, data_dir="."): extracts features from PDB files and merges with classification labels.
	
import os
import sys
import pandas as pd
from Bio.PDB import *
import numpy as np

######### Obtains a dictionary with the properties of the atom names #########
def read_types(data_dir):
    """
    Reads atom types and their properties from a CSV file.

    Args:
        data_dir (str): The directory containing the atom types CSV file.

    Returns:
        dict: A dictionary where keys are atom names and values are lists of properties.
    """
    types = {}
    with open(os.path.join(data_dir,"atom_types.csv"), "r") as in_file:
        for line in in_file:
            record = line.strip().split(',')
            atomName = record[0] + '_' + record[1]
            if len(record) < 3:
                continue
            else:
                types[atomName] = record[2:]
    return types 

######### Obtains a unique index for a given residue in a PDB file #########
def get_residue_index(residue):
    """
    Constructs a unique index for a given residue in a PDB file.

    Args:
        residue (Bio.PDB.Residue): The residue for which the index is to be constructed.

    Returns:
        str: The unique index of the residue in the format 'pdb_id_chain_resName_resNumber'.
    """
    pdb_id = residue.get_full_id()[0]
    chain = residue.get_parent().get_id()
    name = residue.get_resname()
    num = str(residue.get_id()[1])

    return pdb_id + '_' + chain + '_' + name + '_' + num


######### Computes properties of residuals of a given PDB file #########
def compute_properties(data_dir, pdb_file):
    """
    Computes properties for a given PDB file.

    Args:
        data_dir (str): The directory containing the atom types CSV file.
        pdb_file (str): Path to the PDB file.

    Returns:
        pandas.DataFrame: DataFrame containing the computed properties.
    """
    # Parse PDB file
    parser = PDBParser()
    basename = os.path.basename(pdb_file)
    pdb_id = os.path.splitext(basename)[0]
    structure = parser.get_structure(pdb_id, pdb_file)

    # Read atom types
    types = read_types(data_dir)

    # Initialize property matrix
    matrix = pd.DataFrame()
    matrix.index.name = 'res_name'
    # Iterate over residues
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.get_id()[0] == ' ' and is_aa(residue):  # Check if it's a standard amino acid residue
                    residue_name = get_residue_index(residue)
                    matrix.loc[residue_name, ['DON','HPB','ACP','NEG','ARM','POS']] = 0
                    for atom in residue:
                        atom_name = residue.get_resname() + '_' + atom.get_name()
                        if atom_name in types:
                            for prop in ['DON', 'HPB', 'ACP', 'NEG', 'ARM', 'POS']:
                                if prop in types[atom_name]:
                                    matrix.loc[residue_name, prop] += 1

    return matrix.astype(int)


######### Finds neighboring residues for each residue in a PDB structure within a specified radius #########
def find_neighbors(pdb_file, radius=6.0):
    """
    Find neighboring residues for each residue in a PDB structure within a specified radius.

    Args:
        pdb_file (str): Path to the PDB file.
        radius: Distance cutoff for defining neighbors, default is 6.0 Å.
    
    Returns: 
        A dictionary where keys are residue indices and values are sets of neighboring residue indices.
    """

    # Create the parser and load the structure from the PDB file
    parser = PDBParser()
    basename = os.path.basename(pdb_file)
    pdb_id = os.path.splitext(basename)[0]
    structure = parser.get_structure(pdb_id, pdb_file)

    # Create a NeighborSearch object with all atoms in the structure
    atoms = []
    for model in structure:
        for atom in model.get_atoms():
            if atom.get_parent().get_id()[0] == " ":  # Only standard residues
                atoms.append(atom)
    ns = NeighborSearch(atoms)

    neighbors_dict = {}

    # Iterate over each residue in the structure
    for chain in structure.get_chains():
        for residue in chain:
            if residue.get_id()[0] == ' ':  # Only standard residues
                # Set to store neighbors of the current residue
                residue_neighbors = set()

                # Search for neighbors within a specified radius of the current residue
                for atom in residue.get_atoms():
                    neighbors = ns.search(atom.get_coord(), radius)
                    for neighbor in neighbors:
                        # Add neighbors belonging to a different residue to the set
                        if neighbor.get_parent() != residue and neighbor.get_parent().get_id()[0] == " ":
                            neighbor_residue = neighbor.get_parent()
                            neighbor_index = get_residue_index(neighbor_residue)
                            residue_neighbors.add(neighbor_index)

                # Key for the current residue
                residue_index = get_residue_index(residue)

                # Store the set of neighbors of the current residue in the dictionary
                neighbors_dict[residue_index] = list(residue_neighbors)

    return neighbors_dict


######### Weights the properties of each residue by the average of its neighbors #########

def weighted_average(df, neighbors_result):
    # New DataFrame for weighted averages
    weighted_df = pd.DataFrame(index=df.index, columns=df.columns)  
    
    # Iterate through each residue and its neighbors
    for residue, neighbors in neighbors_result.items():
        # Check if residue exists in DataFrame
        if residue in df.index:
            # Calculate mean of neighbors' properties
            neighbor_values = df.loc[neighbors].mean(axis=0)  
            
            # Calculate weighted average and assign it to the corresponding residue
            weighted_df.loc[residue] = (df.loc[residue] + neighbor_values) / 2  
    df=weighted_df
    return df

######### Extract PDB features from PDB file #########
def ExtractPDBFeatures(pdb_file):
    '''
    Extract the main features of PDB files
        - Residue name
        - Relative distance to C-ter
        - Relative distance to N-ter
    Args: 
        pdb_file (str): Path to the PDB file.
    
    Returns: 
        DataFrame with main features for each residue.
        Rows represent residues indexed by 'get_residue_index(residue)'.
        Columns represent DSSP features.
    '''
    parser = PDBParser()
    basename = os.path.basename(pdb_file)
    pdb_id = os.path.splitext(basename)[0]
    structure = parser.get_structure(pdb_id, pdb_file)
    model = structure[0]

    pdb_data = []
    
    # For each chain in the structure obtain all features
    for chain in model:   
        sequence = []
        res_index = 0

        # 1.- Obtain the sequence
        for residue in chain:
            if residue.get_id()[0] == ' ' and is_aa(residue): 
                sequence.append(residue.get_resname())

        # 2.- Obtain the features
        for residue in chain:
            if residue.get_id()[0] == ' ' and is_aa(residue):
                Seq_len = len(sequence)
                Cter_dist = res_index / (Seq_len - 1)
                Nter_dist = 1 - Cter_dist
                res_index += 1
                pdb_data.append((get_residue_index(residue), residue.get_resname(), Cter_dist, Nter_dist))
    
    # Dataframe
    df = pd.DataFrame(pdb_data, columns=['res_name', 'Residue name', 'Cter_dist', 'Nter_dist'])
    return df

######### DSSP features #########
def extract_dssp_features(pdb_file):
    """
    Extract DSSP features from a PDB file.
            - secondary_structure
            - asa
            - phi_angle
            - psi_angle
            - nh_o_1_relidx
            - nh_o_1_energy
            - o_nh_1_relidx
            - o_nh_1_energy
            - nh_o_2_relidx
            - nh_o_2_energy
    Args:
        pdb_file (str): Path to the PDB file.

    Returns:
        DataFrame containing DSSP features for each residue.
        Rows represent residues indexed by 'get_residue_index(residue)',
        Columns represent DSSP features.
    """
    # Parse the PDB file
    parser = PDBParser()
    basename = os.path.basename(pdb_file)
    pdb_id = os.path.splitext(basename)[0]
    structure = parser.get_structure(pdb_id, pdb_file)
    
    # Initialize DSSP object
    model = structure[0]
    dssp = DSSP(model, pdb_file)
    # Initialize empty lists
    dssp_features = []
    res_index = []

# Loop over chains and residues to get residue indices
    for chain in model:
        for residue in chain:
            residue_index = get_residue_index(residue)
            res_index.append(residue_index)

# Extract DSSP features
    residues = list(dssp)
    for residue, residue_index in zip(residues, res_index):  # Iterate over both residues and indices
        secondary_structure = residue[2]
        asa = residue[3]
        phi_angle = residue[4]
        psi_angle = residue[5]
        nh_o_1_relidx = residue[6]
        nh_o_1_energy = residue[7]
        o_nh_1_relidx = residue[8]
        o_nh_1_energy = residue[9]
        nh_o_2_relidx = residue[10]
        nh_o_2_energy = residue[11]

    # Append to the list of DSSP features
        dssp_features.append((
            residue_index, secondary_structure, asa, phi_angle, psi_angle,
            nh_o_1_relidx, nh_o_1_energy,
            o_nh_1_relidx, o_nh_1_energy,
            nh_o_2_relidx, nh_o_2_energy
        ))

# Convert to DataFrame
    columns = ['res_name', 'SS', 'ASA', 'Phi Angle', 'Psi Angle', 'NH->O_1_relidx', 'NH->O_1_energy', 
            'O->NH_1_relidx', 'O->NH_1_energy', 'NH->O_2_relidx', 'NH->O_2_energy']
    df = pd.DataFrame(dssp_features, columns=columns)
    df.set_index('res_name', inplace=True)  # Set residue index as DataFrame index

    return df


######## MERGE DATAFRAMES ########
def merge_dataframes(df_list, on_column):
    """
    Merge multiple DataFrames on a specified column, considering only the common rows.

    Args:
        df_list (list): List of DataFrames to merge.
        on_column (str): The name of the column on which to merge the DataFrames.

    Returns:
        DataFrame: The merged DataFrame containing only the common rows.
    """

    # Merge DataFrames
    merged_df = df_list[0]  # Tomar el primer DataFrame como base
    for df in df_list[1:]:
        merged_df = pd.merge(merged_df, df, on=on_column, how='inner')

    return merged_df
    
######### EXTRACT ALL FEATURES ########
def extract_features(pdb_dir, labels_dir, data_dir="."):
    """
    Extracts features from PDB files and merges with classification labels.

    Args:
        pdb_dir (str): Directory containing the PDB files.
        labels_dir (str): Directory containing the CSV files with classification labels.
        data_dir (str): Directory containing the required atom_types.csv file.

    Returns:
        pandas.DataFrame: DataFrame with extracted features and merged classification labels.
    """
    # Check if atom_types.csv exists in the data_dir
    if "atom_types.csv" not in os.listdir(data_dir):
        print("Warning: atom_types.csv file not found in the specified data directory.")
        print("Please make sure that atom_types.csv is in the current directory or specify the correct data directory.")
    
    merged_dfs = []
    sys.stderr.write("###Extracting features... please be patient\n")
    for pdb_file in os.listdir(pdb_dir):
        if pdb_file.endswith('.pdb'):
            pdb_id = os.path.splitext(pdb_file)[0]
            labels_file = os.path.join(labels_dir, f"{pdb_id}.csv")
            
            if os.path.exists(labels_file):
                parser = PDBParser()
                structure = parser.get_structure(pdb_id, os.path.join(pdb_dir, pdb_file))

                features1 = ExtractPDBFeatures(os.path.join(pdb_dir, pdb_file))
                features2 = extract_dssp_features(os.path.join(pdb_dir, pdb_file))
                compute_prop_df= compute_properties(data_dir, os.path.join(pdb_dir, pdb_file))
                neighbors_results = find_neighbors(os.path.join(pdb_dir, pdb_file))
                features3 = weighted_average(compute_prop_df, neighbors_results)

                all_features = merge_dataframes([features1, features2, features3], 'res_name')
                
                labels_df = pd.read_csv(labels_file, usecols=['res_name', 'class'], index_col='res_name')

                merged_df = merge_dataframes([labels_df, all_features], 'res_name')

                merged_dfs.append(merged_df)

    sys.stderr.write("###Features correctly extracted\n")

    if merged_dfs:
        final_df = pd.concat(merged_dfs)
    else:
        final_df = pd.DataFrame()

    return final_df
