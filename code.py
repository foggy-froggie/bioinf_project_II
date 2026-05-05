# %% [markdown]
# https://tdcommons.ai/single_pred_tasks/adme/#solubility-aqsoldb
# Aqeuous solubility measures a drug's ability to dissolve in water. Poor water solubility could lead to slow drug absorptions, inadequate bioavailablity and even induce toxicity. More than 40% of new chemical entities are not soluble. 

# %% TASK 1
import pandas as pd
from tdc.single_pred import ADME
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator

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

# %% TASK 4

def assign_morgan_fingerprints(df):
    """
    Assigns Morgan fingerprints with different radius and length combinations.
    Creates 9 versions: radius 1, 2, 3 and length 512, 1024, 2048.
    
    Args:
        df (pd.DataFrame): DataFrame with parsed molecules including 'mol' column
        
    Returns:
        pd.DataFrame: New dataframe with original columns plus 9 Morgan fingerprint columns
    """
    if len(df) == 0:
        print("No molecules to process.")
        return pd.DataFrame()
    
    if 'mol' not in df.columns:
        print("Error: 'mol' column not found in dataframe.")
        return df
    
    # Create a copy of the dataframe
    new_df = df.copy()
    
    # Define radius and length combinations
    radii = [1, 2, 3]
    lengths = [512, 1024, 2048]
    
    print(f"\nGenerating Morgan fingerprints for {len(df)} molecules...")
    print(f"  Radius options: {radii}")
    print(f"  Length options: {lengths}")
    print(f"  Total combinations: {len(radii) * len(lengths)}")
    
    # Generate fingerprints for each combination
    for radius in radii:
        for length in lengths:
            column_name = f'morgan_r{radius}_l{length}'
            fingerprints = []
            # Create generator once per combination (can be reused for all molecules)
            fpgen = rdFingerprintGenerator.GetMorganGenerator(radius, length)
            
            for idx, row in df.iterrows():
                mol = row['mol']
                if mol is None:
                    fingerprints.append(None)
                else:
                    try:
                        # Generate Morgan fingerprint using rdFingerprintGenerator
                        fp = fpgen.GetFingerprint(mol)
                        # Convert to list for storage in dataframe
                        fingerprints.append(list(fp))
                    except Exception as e:
                        print(f"Warning: Failed to generate fingerprint for molecule {idx}: {e}")
                        fingerprints.append(None)
            
            new_df[column_name] = fingerprints
            print(f"  Generated: {column_name}")
    
    print(f"\nMorgan fingerprint assignment complete!")
    print(f"  New dataframe shape: {new_df.shape}")
    print(f"  New columns added: {len(radii) * len(lengths)}")
    
    return new_df

# Assign fingerprints
print("Assigning Morgan fingerprints...")
fingerprint_df = assign_morgan_fingerprints(df)

print(f"\nFingerprint dataframe shape: {fingerprint_df.shape}")
print(f"Fingerprint dataframe columns: {fingerprint_df.columns.tolist()}")
print(f"\nFirst few rows (showing non-fingerprint columns):")
# Display without the mol and fingerprint columns
display_cols = [col for col in fingerprint_df.columns 
               if col != 'mol' and not col.startswith('morgan_')]
fingerprint_df[display_cols].head()

# %%
