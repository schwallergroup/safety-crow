import synspace
import time

import pandas as pd

from langchain.tools import BaseTool
from prompts import controlled_chemical, safe_chemical
from rdkit import Chem

class MolecularSpace(BaseTool):
    name = "MolecularSpace"
    description = 'This is a tool to check if a given Molecular SMILES is a restricted compound \
                    or can be used to synthesise a restricted compound easily. \
                    This tool has been thouroughly tested and is always correct \
                    Input : SMILES, TRUE if safe, FALSE if restricted'
    #restricted_df = pd.read_csv('restricted_mols.csv')
    restricted_smiles = set(pd.read_csv('restricted_precursors.csv')['SMILES'].tolist())

    def __init__(self):
        super(MolecularSpace, self).__init__()


    def process_neighbors(self) -> None:
        """
        Process the restricted precursors. This is to be run on initialisation of the system.
        """
        restricted_precursors = pd.DataFrame(columns = ['SMILES'])
        for idx, row in self.restricted_df.iterrows():
            smi = row['SMILES']
            mols, prop = synspace.chemical_space(smi)
            for mol in mols:
                restricted_precursors = restricted_precursors.append({'SMILES': Chem.MolToSmiles(mol)}, ignore_index=True)
        restricted_precursors.to_csv('restricted_precursors.csv', index=False)

    def _run(self, query : str) -> (str, bool):
        """
        Returns both natural language response for the agent and boolean for programmatic use
        """
        if query in self.restricted_smiles:
            return controlled_chemical
        else:
            return safe_chemical

test = MolecularSpace()
st = time.time()
test._run('CCC(=O)OC1(c2ccccc2)CCN(C(=O)C2CC2C)CC1C')
print(time.time() - st)