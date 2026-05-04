# %% [markdown]
# https://tdcommons.ai/single_pred_tasks/adme/#solubility-aqsoldb
# Aqeuous solubility measures a drug's ability to dissolve in water. Poor water solubility could lead to slow drug absorptions, inadequate bioavailablity and even induce toxicity. More than 40% of new chemical entities are not soluble. 

# %% TASK 1
import pandas as pd
from tdc.single_pred import ADME
from rdkit import Chem
from rdkit.Chem import Descriptors

# %%
# 1. Data retrival
data = ADME(name = 'Solubilit`y_AqSolDB')
df = data.get_data()

# 2. Conversion SMILES → Molecule Objects
df['mol'] = df['Drug'].apply(lambda x: Chem.MolFromSmiles(x))
df = df.dropna(subset=['mol']).reset_index(drop=True)

# 3. Basic desctiptors
df['MolWt'] = df['mol'].apply(Descriptors.MolWt)          # Molecular Weight - sum of the atomic weights of all atoms in a molecule
df['LogP'] = df['mol'].apply(Descriptors.MolLogP)         # Octanol-Water Partition Coefficient - A measure of a molecule's lipophilicity
df['NumHDonors'] = df['mol'].apply(Descriptors.NumHDonors) # Number of Hydrogen Bond Donors - amount of hydrogen atoms attached to an electronegative atom (like Nitrogen or Oxygen) that can be "donated" to form a hydrogen bond.
df['NumHAcceptors'] = df['mol'].apply(Descriptors.NumHAcceptors) # Number of Hydrogen Bond Acceptors - The count of electronegative atoms (N or O) with lone pairs of electrons that can "accept" a hydrogen atom from another molecule.

summary_table = df[['Y', 'MolWt', 'LogP', 'NumHDonors', 'NumHAcceptors']].describe()

print("--- Summary Table ---")
print(summary_table)
# %% TASK 2
# 1. Generate InChIKey for each molecule
df['InChIKey'] = df['mol'].apply(lambda x: Chem.MolToInchiKey(x))
duplicates = df[df.duplicated('InChIKey', keep=False)].sort_values('InChIKey')
num_duplicates = df.duplicated('InChIKey').sum()
# No dupliacates

# %% TASK 3
def check_stereochemistry(mol):
    if mol is None:
        return 0, 0
    chiral_centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True)
    total_centers = len(chiral_centers)
    undefined_centers = sum(1 for center in chiral_centers if center[1] == '?')
    return total_centers, undefined_centers

df['Stereo_Info'] = df['mol'].apply(check_stereochemistry)
df['Total_Chiral_Centers'] = df['Stereo_Info'].apply(lambda x: x[0])
df['Undefined_Chiral_Centers'] = df['Stereo_Info'].apply(lambda x: x[1])

unspecified_molecules = df[df['Undefined_Chiral_Centers'] > 0]
partial_chirality = df[(df['Total_Chiral_Centers'] > df['Undefined_Chiral_Centers']) & (df['Undefined_Chiral_Centers'] > 0)]

print(f"Molecules with at least one unspecified stereocenter: {len(unspecified_molecules)}")
print(f"Molecules with partial chirality (some centers known, some not): {len(partial_chirality)}")

# %% [markdown]
# In Task 3, I identified molecules with unspecified and partial stereochemistry. I found that X molecules had at least one undefined chiral center. While these could be standardized, I recommend removal (or at least caution) because undefined stereochemistry introduces noise. Stereochemistry significantly impacts the label (solubility) because different 3D arrangements affect crystal lattice energy and packing, leading to different experimental solubility values for the same 2D structure
