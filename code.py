# %% [markdown]
# https://tdcommons.ai/single_pred_tasks/adme/#solubility-aqsoldb
# Aqeuous solubility measures a drug's ability to dissolve in water. Poor water solubility could lead to slow drug absorptions, inadequate bioavailablity and even induce toxicity. More than 40% of new chemical entities are not soluble. 

# %% TASK 1
import pandas as pd
from tdc.single_pred import ADME
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from rdkit.DataStructs import BulkTanimotoSimilarity
from tdc.utils.split import create_scaffold_split
import umap
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

# %%
# 1. Data retrival
data = ADME(name = 'Solubilit`y_AqSolDB')

def merge_split(split):
    train, test, valid = split["train"], split["test"], split["valid"]
    train["split"] = "train"
    test["split"] = "test"
    valid["split"] = "valid"
    return pd.concat([train, test, valid])

# df = merge_split(data.get_split())
df = merge_split(create_scaffold_split(df, seed=42, frac=[0.7, 0.1, 0.2], entity=data.entity1_name))

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
assert int(num_duplicates) == 0
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
# Unspecified Stereocenters: Out of the dataset, 2169 molecules were identified as having at least one unspecified chiral center, meaning their exact 3D structural orientation is unknown in the dataset.
# Partial Chirality: Among those, 68 molecules exhibit partial chirality, where some stereocenters are clearly defined while others remain ambiguous.
# Stereochemistry can affect solubility as 3D geometry influences crystal packing and water interactions.
# Therefore, molecules with unspecified stereocenters may introduce unclarity in the target variable.
# However for our model we use Morgan fingerprints without chirality encoding, which are based on 2D molecular topology and do not capture stereochemical information.
# Hence we keep all the molecules keeping the size of the originial dataset, while understanding that they may introduce some level of noise.

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
                        fingerprints.append(fp)
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
reducer = umap.UMAP()
column_name = f'morgan_r{2}_l{512}'
features = np.array(list(fingerprint_df[column_name]))
embedding = reducer.fit_transform(features)
embedding.shape
fingerprint_df["embedding_x"] = embedding[:, 0]
fingerprint_df["embedding_y"] = embedding[:, 1]

# %%
sns.set()

# %%
ax = sns.scatterplot(
    data=fingerprint_df,
    x="embedding_x",
    y="embedding_y",
    hue="Y",
    s=5,
    palette="RdBu",
)
norm = plt.Normalize(df.Y.min(), df.Y.max())
sm = plt.cm.ScalarMappable(cmap="RdBu", norm=norm)
sm.set_array([])
ax.get_legend().remove()
ax.figure.colorbar(sm, label="Y", ax=ax)
# %% [markdown]
# We can see two large clusters, they are close toghether, but visually easily separable.
# Unfortunately, they don't cleanly separate based on labels.
# Most points are blue (high label value), there are a few red regions (low label value), but they are mostly mixed with blue and hard to separate.
# This indicates that training a good model might not be easy.

# %%
sns.scatterplot(
    data=fingerprint_df,
    x="embedding_x",
    y="embedding_y",
    hue="split",
    s=5,
    alpha=1,
)
# %% [markdown]
# Datasets are thoroughly mixed, as expected with a random split.
# This isn't good for training, we need a different split method to separate train and test datasets.

# %%
fdf = fingerprint_df

# %%
train = fdf.loc[fdf["split"] == "train", column_name].to_list()
test = fdf.loc[fdf["split"] == "test", column_name].to_list()
train_train_similarity = np.array([BulkTanimotoSimilarity(e, train) for e in train])
train_test_similarity = np.array([BulkTanimotoSimilarity(e, test) for e in train])
test_test_similarity = np.array([BulkTanimotoSimilarity(e, test) for e in test])

# %%
plt.hist(train_test_similarity.max(axis=0))
# %% [markdown]
# We can see there's a lot of high similarity values and there's not much difference between intra- and inter-dataset similarity distributions.
# That's bad. We need a better split.

# %%
# line histogram insted of KDE plot, because KDE is very slow for this large data set
def plot_hist(data, bins=20, range=(0, 1), density=True, **kwargs):
    hist, bin_edges = np.histogram(data, bins=bins, range=range, density=density, **kwargs)
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
    plt.plot(bin_centers, hist)
plot_hist(train_train_similarity)
plot_hist(train_test_similarity)
plot_hist(test_test_similarity)
# %% [markdown]
# This is even worse.
# Density plots/histograms of train-train, train-test and test-test similarities are identical.
# This means the data in train and test datasets is basically identically distributed and the test dataset doesn't represent any genuinely new chemistry.
# We need a better split.

# %%
