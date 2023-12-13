'''
preprocessing.py

ABOUT:
- 
'''

import os
import time
import numpy as np
import pandas as pd
import pymol

from rdkit import Chem
from log.logger import Logger

# from rdkit import RDLogger
# RDLogger.DisableLog('rdApp.*')

np.set_printoptions(threshold=np.inf)


'''
label_chain_around_ligand:
    about:
        label pdb_chain_id druggable cavity regions around a ligand
    params:
        pdb_chain_id = 4-character PDB ID with chain identifier
        lig_id_list = list of 1- to 3-character ligand het_ids
        het_id_list = list of 1- to 3-character other het_ids
        complex_dir = file path to folder with protein chain, labels, etc.
        working_dir = working directory
        logger = preprocessing logger
        radius_pocket = cutoff radius for determining pocket atoms around ligand in Angstroms
        expand_pockets_by_res = boolean whether entire parent residue is included in pocket region or not
        cut_out_pockets = boolean whether individual labeled pocket(s) or entire labeled protein .pdb is saved or not
        include_ligand = boolean whether ligand is included or removed
    returns:
        protein_path = file path to labeled protein or pocket .pdb
        labels_path = file path to atom labels (b-factors)
        exit_code = exit code if preprocessing completed successfully
'''
def label_chain_around_ligand(pdb_path, pdb_chain_id, lig_id_list, het_id_list,
                              complex_dir, working_dir, logger, radius_pocket,
                              expand_pockets_by_res, cut_out_pockets, include_ligand):

    exit_code = 0
    pdb_id = pdb_chain_id.split('_')[0].upper()
    chain_id = pdb_chain_id.split('_')[1].upper()
    
    # set up PyMOL environment; sanitize protein .pdb; set all b-factors to 0.00
    pymol.cmd.feedback('disable', 'all', 'actions')
    pymol.cmd.feedback('disable', 'all', 'results')

    # protein_dir = os.path.join(complex_dir, f"{pdb_id}_rcsb.pdb")
    # pymol.cmd.load(os.path.join(working_dir, protein_dir))
    pymol.cmd.load(os.path.join(working_dir, pdb_path))

    pymol.cmd.remove(f'not chain {chain_id.upper()}') # remove all other chains
    pymol.cmd.remove('hydrogen')
    pymol.cmd.remove("not alt ''+A") # remove all alternate atom records

    pymol.cmd.alter('all', 'b=0.00') # set all b-factors to 0.00 initially

    # identify which lig_ids are present in the pdb_chain_id .pdb
    lig_id_present = []
    for lig_id in lig_id_list:

        lig_id = lig_id.upper()

        # check that lig_id identifier is valid (1-3 alphanumeric characters)
        if not lig_id.isalnum() and len(lig_id) not in [1,2,3]:
            logger.error(f"[{pdb_chain_id}]: improperly formatted lig_id [{lig_id}]; must be 1-3 alphanumeric characters")
            exit_code = 1
            return None, None, exit_code

        if pymol.cmd.select(f'resn {lig_id}'):
            lig_id_present.append(lig_id)

    if not lig_id_present:
        logger.warning(f"[{pdb_chain_id}]: no druggable regions labeled")
        pymol.cmd.delete('all')
        exit_code = 1
        return None, None, exit_code

    # concatenate lig_ids and het_ids, and remove all non-polymer, non-lig_id_present, and non-het_id_list from .pdb
    keep_str = 'polymer + '
    lig_id_present_str = 'resn ' + ' + resn '.join(lig_id_present)
    keep_str += lig_id_present_str

    if het_id_list:
        keep_str += ' + resn '
        keep_str += ' + resn '.join(het_id_list)

    pymol.cmd.remove(f'not ({keep_str})')

    # label regions around lig_ids within radius_pocket
    if lig_id_present:
        if expand_pockets_by_res == True:
            pymol.cmd.select('druggable', f'byres {lig_id_present_str} around {radius_pocket}')

        else:
            pymol.cmd.select('druggable', f'{lig_id_present_str} around {radius_pocket}')

        pymol.cmd.alter('druggable', 'b=1.00')
        logger.info(f"[{pdb_chain_id}]: atoms around ligand(s) {lig_id_present} labeled [1, DRUGGABLE]")

    # export labeled protein or pocket structure
    pymol.cmd.set('pdb_conect_all', 'on') # include CONECT records for all atoms
    pymol.cmd.set('pdb_conect_nodup', '0') # include duplicate CONECT records for multiple bonds
    
    if cut_out_pockets == True: # NEED TO HANDLE CASE WHERE ONE REGION CONTAINS MULTIPLE LIGANDS OR MULTIPLE OF SAME LIGAND IN PROTEIN; multiple ligands could be due to alternate positions
        
        protein_path = os.path.join(complex_dir, f"{pdb_chain_id}_pockets")
        if not os.path.exists(os.path.join(working_dir, protein_path)):
            os.mkdir(os.path.join(working_dir, protein_path))

        labels_path = os.path.join(complex_dir, f"{pdb_chain_id}_labels")
        if not os.path.exists(os.path.join(working_dir, labels_path)):
            os.mkdir(os.path.join(working_dir, labels_path))

        for lig_id in lig_id_present:
            if expand_pockets_by_res == True:
                if include_ligand == True:
                    pymol.cmd.select('pocket', f'(byres resn {lig_id} around {radius_pocket}) or resn {lig_id}')
                elif include_ligand == False:
                    pymol.cmd.select('pocket', f'(byres resn {lig_id} around {radius_pocket}) and not ({lig_id_present_str})')
            elif expand_pockets_by_res == False:
                if include_ligand == True:
                    pymol.cmd.select('pocket', f'(resn {lig_id} around {radius_pocket}) or resn {lig_id}')
                elif include_ligand == False:
                    pymol.cmd.select('pocket', f'(resn {lig_id} around {radius_pocket}) and not ({lig_id_present_str})')

            labeled_pocket_path = os.path.join(working_dir, protein_path, f"{lig_id}_pocket.pdb")
            pymol.cmd.save(labeled_pocket_path, 'pocket')
        
            pymolspace = {'bfactors': []}
            pymol.cmd.iterate('pocket', 'bfactors.append(b)', space=pymolspace) # obtain b-factors from structure
            b_factors = np.array(pymolspace['bfactors'], dtype='int')

            pocket_labels_path = os.path.join(working_dir, labels_path, f"{lig_id}_labels.npy")
            np.save(pocket_labels_path, b_factors)

            # logger.info(f"[{pdb_chain_id}]: labeled [{lig_id}] pocket .pdb and labels .npy saved")

    if cut_out_pockets == False:
        
        # remove lig_ids according to include_ligand
        if include_ligand == False:
            pymol.cmd.remove(f'{lig_id_present_str}')

        labeled_protein_chain_path = os.path.join(complex_dir, f"{pdb_chain_id}_protein.pdb")
        pymol.cmd.save(os.path.join(working_dir, labeled_protein_chain_path), 'all')

        pymolspace = {'bfactors': []}
        pymol.cmd.iterate('all', 'bfactors.append(b)', space=pymolspace) # obtain b-factors from structure
        b_factors = np.array(pymolspace['bfactors'], dtype='int')

        protein_labels_path = os.path.join(complex_dir, f"{pdb_chain_id}_labels.npy")
        np.save(os.path.join(working_dir, protein_labels_path), b_factors)

        # logger.info(f"[{pdb_chain_id}]: labeled protein chain .pdb and labels .npy saved")

        protein_path = labeled_protein_chain_path
        labels_path = protein_labels_path

    # reset PyMOL
    pymol.cmd.delete('all')

    # check that RDKit can load labeled protein without error
    try:
        protein_chain = Chem.rdmolfiles.MolFromPDBFile(os.path.join(working_dir, protein_path), sanitize=True, removeHs=True, proximityBonding=False)
        protein_chain.GetAtoms()

    except Exception as e:
        logger.error(f"[{pdb_chain_id}]: RDKit error reading labeled protein or pocket .pdb [{protein_path}]; see RDKit Logger; error message [{e}]")
        exit_code = 1
        return None, None, exit_code
    
    return protein_path, labels_path, exit_code


'''
preprocessing:
    about:
        creates labeled protein/pocket .pdb and label .npy for each protein chain in pdb_chain_id_list
    params:
        pro_data_path = file path to .tsv of pdb_chain_ids (to sample from) for pdb_chain_id_list
        lig_data_path = file path to .tsv of ligand het_ids for lig_id_list
        het_data_path = file path to .tsv of other het_ids for het_id_list



        data_dir = file path to data folder
        working_dir = working directory
        sample_size = pdb_data sample size
        radius_pocket = cutoff radius for determining pocket atoms around ligand in Angstroms
        expand_pockets_by_res = boolean whether entire parent residue is included in pocket region or not
        cut_out_pockets = boolean whether individual labeled pocket(s) or entire labeled protein .pdb is saved or not
        include_ligand = boolean whether ligand is included or removed
    returns:
        data_df = dataframe of all processed data paths and parameters
        data_path = file path to data_df as .csv
'''
def preprocessing(pro_data_path, lig_data_path, het_data_path, pdb_data_dir, data_dir, working_dir, sample_size, radius_pocket=5,
                  expand_pockets_by_res=False, cut_out_pockets=False, include_ligand=True):
    
    # log parameter information
    logger = Logger('preprocessing.log')

    logger.critical("PREPROCESSING PARAMETERS")
    logger.info(f"pro_data_path = {pro_data_path}")
    logger.info(f"lig_data_path = {lig_data_path}")
    logger.info(f"het_data_path = {het_data_path}")

    # get pdb_chain_id_list = list of 4-character PDB IDs with chain identifiers for data set
    pro_data_path = os.path.join(working_dir, pro_data_path)
    if os.path.exists(pro_data_path):
        pdb_df = pd.read_csv(pro_data_path, sep='\t', header=0)
        pdb_data_randomized = pdb_df.sample(n=sample_size, axis=0)
        pdb_chain_id_list = list(map(str, pdb_data_randomized.iloc[:, 1])) # MAY NEED TO MODIFY
    else:
        pdb_chain_id_list = []

    # get lig_id_list = list of 1- to 3-character ligand het_ids
    lig_data_path = os.path.join(working_dir, lig_data_path)
    if os.path.exists(lig_data_path):
        lig_df = pd.read_csv(lig_data_path, sep='\t', header=0)
        lig_id_list = list(map(str, list(lig_df['HET_ID']))) # MAY NEED TO MODIFY
    else:
        lig_id_list = []

    # get het_id_list = list of 1- to 3-character other het_ids
    het_data_path = os.path.join(working_dir, het_data_path)
    if os.path.exists(het_data_path):
        het_df = pd.read_csv(het_data_path, sep='\t', header=0)
        lig_id_list = list(map(str, list(lig_df['HET_ID']))) # MAY NEED TO MODIFY
    else:
        het_id_list = []

    # continue log parameter information
    logger.info(f"sample_size = {sample_size}")
    logger.info(f"data_dir = {data_dir}")
    logger.info(f"radius_pocket = {radius_pocket}")
    logger.info(f"expand_pockets_by_res = {expand_pockets_by_res}")
    logger.info(f"cut_out_pockets = {cut_out_pockets}")
    logger.info(f"include_ligand = {include_ligand}")

    logger.info(f"LEN pdb_chain_id_list = {len(pdb_chain_id_list)}")
    logger.info(f"LEN lig_id_list = {len(lig_id_list)}")
    logger.info(f"LEN het_id_list = {len(het_id_list)} ")

    logger.critical("PREPROCESSING START")

    # start preprocessing
    if not os.path.exists(os.path.join(working_dir, data_dir)):
        os.mkdir(os.path.join(working_dir, data_dir))

    preprocessing_data = []
    count = 1

    for pdb_chain_id in pdb_chain_id_list:
        start = time.time()
        
        pdb_chain_id = pdb_chain_id.upper()

        print(f"\n[{pdb_chain_id}]: {count} of {len(pdb_chain_id_list)} protein chains")
        count += 1

        complex_dir = os.path.join(data_dir, pdb_chain_id)
        if not os.path.exists(os.path.join(working_dir, complex_dir)):
            os.mkdir(os.path.join(working_dir, complex_dir))

        pdb_id = pdb_chain_id.split('_')[0].upper()
        pdb_path = os.path.join(pdb_data_dir, f"{pdb_id.lower()}.pdb")
        
        protein_path, labels_path, exit_code = label_chain_around_ligand(pdb_path, pdb_chain_id, lig_id_list, het_id_list,
                                                                         complex_dir, working_dir, logger, radius_pocket,
                                                                         expand_pockets_by_res, cut_out_pockets, include_ligand)

        if exit_code == 0: # if labeling is completed correctly
            preprocessing_data.append([pdb_chain_id, complex_dir, pdb_path,
                                       protein_path, labels_path, radius_pocket,
                                       expand_pockets_by_res, cut_out_pockets, include_ligand])
            
        else: # if error occurs in labeling
            logger.error(f"[{pdb_chain_id}]: error in label_chain_around_ligand; ignoring this .pdb")
        
        print(f"[{pdb_chain_id}]: completed in {np.round(time.time()-start,4)} seconds")
    
    data_df = pd.DataFrame(preprocessing_data, columns=['pdb_chain_id','complex_dir','pdb_path','protein_path','labels_path','radius_pocket','expand_pockets_by_res','cut_out_pockets','include_ligand'])

    logger.critical("PREPROCESSING END")

    # save dataframe as CSV
    data_path = os.path.join(data_dir, f"data_paths.csv")
    data_df.to_csv(data_path)
    logger.info(f"data_path = {data_path}")
    logger.info(f"LEN data_df = {len(data_df)}")

    return data_df, data_path


'''
main:
    about:
        runs preprocessing pipeline
    params:
        pdb_data_path = file path to .tsv of pdb_chain_ids (to sample from) for pdb_chain_id_list
        lig_data_path = file path to .tsv of ligand het_ids for lig_id_list
        het_data_path = file path to .tsv of other het_ids for het_id_list
        data_dir = file path to data folder
        working_dir = working directory
        sample_size = pdb_data sample size
        radius_pocket = cutoff radius for determining pocket atoms around ligand in Angstroms
        expand_pockets_by_res = boolean whether entire parent residue is included in pocket region or not
        cut_out_pockets = boolean whether individual labeled pocket(s) or entire labeled protein .pdb is saved or not
        include_ligand = boolean whether ligand is included or removed
    returns: none
'''
if __name__ == '__main__':

    pro_data_path = "data/raw/pro/one_per_cluster.tsv"
    lig_data_path = "data/raw/lig/druglike_ligands.tsv"
    het_data_path = "NA"

    pdb_data_dir = "data/pdb"

    data_dir = "data/processed"
    working_dir = os.getcwd()

    # run preprocessing
    preprocessing(pro_data_path, lig_data_path, het_data_path, pdb_data_dir, data_dir, working_dir, sample_size=3600, radius_pocket=5,
                  expand_pockets_by_res=True, cut_out_pockets=False, include_ligand=False)
