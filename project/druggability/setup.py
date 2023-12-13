'''
setup.py

ABOUT:
- 
'''

import os
import requests
import pandas as pd

'''
get_pdb:
    about:
        download pdb_id from RCSB and save to dir
    params:
        pdb_id: 4-character PDB ID (possibly with chain identifier)
        dir: directory to save PDB
    returns:
        name: path to pdb_id
'''
def get_pdb(pdb_id, dir, verbose=False):

    if not os.path.exists(dir):
        os.mkdir(dir)

    data = requests.get(f"http://files.rcsb.org/download/{id.lower()}.pdb")

    path = os.path.join(dir, f"{id}.pdb")
    with open(path, 'wb') as pdb:
        pdb.write(data.content)

    if verbose:
        print(f"[{pdb_id}]: downloaded PDB from RCSB to [{path}]")

    return path


if __name__ == '__main__':

    protein_data_path = r"data/raw/pro/one_per_cluster.tsv"
    pdb_data_dir = r"data/pdb"

    protein_data = pd.read_csv(protein_data_path, sep='\t', header=0)

    # pdb_chain_id_list: list of 4-character PDB IDs with chain identifiers
    pdb_chain_id_list = list(map(str, protein_data.iloc[:, 1])) # MAY NEED TO MODIFY

    for count, pdb_chain_id in enumerate(pdb_chain_id_list):
        try:
            print(f"\n[{pdb_chain_id.upper()}]: {count+1} of {len(pdb_chain_id_list)} protein chains")
            pdb_id = pdb_chain_id.split('_')[0].lower()

            ########### TEMP
            if pdb_id == '7UKY':
                pdb_id = '8TN3'
            elif pdb_id == '8ASQ':
                pdb_id = '8CRF'

            get_pdb(pdb_id, pdb_data_dir)
        except:
            print(f"ERROR: {pdb_chain_id}")
            continue
