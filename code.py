# %% [markdown]
# Source: https://tdcommons.ai/single_pred_tasks/adme/#solubility-aqsoldb
# <br>
# **Aqueous solubility** ($log\ mol/L$) is a property that dictates how a drug is absorbed, its bioavailability, and its potential toxicity.  
# <br>
# The AqSolDB dataset is a collection of aqueous solubility data for 9,980 unique chemical compounds. 
# The dataset contains:
# - Drug_ID
# - SMILES Strings: 2D chemical structures of the molecules.
# - Target ($Y$): Experimental solubility values measured on a logarithmic scale ($log\ mol/L$).
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
import xgboost as xgb
from sklearn.metrics import root_mean_squared_error, mean_squared_error
from scipy.stats import spearmanr

# %%
# 1. Data retrieval
data = ADME(name = 'Solubility_AqSolDB')
df = data.get_data()

# %%
# 2. Create random split
def assign_split(df, col_name, split):
    df = df.set_index("Drug_ID")
    for key in split.keys():
        index = split[key]["Drug_ID"]
        # for some reason there are two keys which are missing in the filtered dataframe after drop_na lol
        index = index[(index != '3-methyl-n-oxidepyridine') & (index != 'n-oxidenicotinic acid')]
        df.loc[index, col_name] = key
    return df.reset_index()

df = assign_split(df, "split_random", data.get_split(method="random"))

# %%
# 3. Conversion SMILES → Molecule Objects
df['mol'] = df['Drug'].apply(lambda x: Chem.MolFromSmiles(x))

# 4. Remove missing data
df = df.dropna(subset=['mol']).reset_index(drop=True)

# %%
# 5. Only after removing missing data, we can create a scaffold split.
# Calling `data.get_split(method="scaffold")` directly results in an internal package error...
df = assign_split(df, "split_scaffold", create_scaffold_split(df, seed=42, frac=[0.7, 0.1, 0.2], entity=data.entity1_name))

# %%
# 6. Basic descriptors
df['MolWt'] = df['mol'].apply(Descriptors.MolWt)          # Molecular Weight - sum of the atomic weights of all atoms in a molecule
df['LogP'] = df['mol'].apply(Descriptors.MolLogP)         # Octanol-Water Partition Coefficient - A measure of a molecule's lipophilicity
df['NumHDonors'] = df['mol'].apply(Descriptors.NumHDonors) # Number of Hydrogen Bond Donors - amount of hydrogen atoms attached to an electronegative atom (like Nitrogen or Oxygen) that can be "donated" to form a hydrogen bond.
df['NumHAcceptors'] = df['mol'].apply(Descriptors.NumHAcceptors) # Number of Hydrogen Bond Acceptors - The count of electronegative atoms (N or O) with lone pairs of electrons that can "accept" a hydrogen atom from another molecule.

summary_table = df[['Y', 'MolWt', 'LogP', 'NumHDonors', 'NumHAcceptors']].describe()

print("--- Summary Table ---")
print(summary_table)
# %% [markdown]
# ### Table analysis
# - Solubility ($Y$): With a mean of -2.89 and a range spanning from -13.17 to 2.14, the dataset covers a massive spectrum of solubility. The standard deviation of 2.37 indicates high variability, which is ideal for training a machine learning model.  
# <br>
# - Molecular Weight (MolWt): The average weight is 266.69 g/mol, which is typical for "small molecule" drugs. However, the maximum value of 5,299 g/mol shows the inclusion of very large, complex structures.
# - LogP (Lipophilicity): The mean LogP of 1.98 suggests that the average molecule in the set is slightly hydrophobic (prefers oil over water). The extreme range (-40.87 to 68.54) reflects an incredibly diverse set of chemical behaviors.  
# - Hydrogen Bond Descriptors: The molecules have an average of 1.11 donors and 3.41 acceptors. These features are critical for solubility because they dictate how well a molecule can bond with water molecules to dissolve

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
fdf = assign_morgan_fingerprints(df)

print(f"\nFingerprint dataframe shape: {fdf.shape}")
print(f"Fingerprint dataframe columns: {fdf.columns.tolist()}")
print(f"\nFirst few rows (showing non-fingerprint columns):")
# Display without the mol and fingerprint columns
display_cols = [col for col in fdf.columns 
               if col != 'mol' and not col.startswith('morgan_')]
fdf[display_cols].head()

# %%
reducer = umap.UMAP()
column_name = f'morgan_r{2}_l{512}'
features = np.array(list(fdf[column_name]))
embedding = reducer.fit_transform(features)
embedding.shape
fdf["embedding_x"] = embedding[:, 0]
fdf["embedding_y"] = embedding[:, 1]

# %%
sns.set()

# %%
ax = sns.scatterplot(
    data=fdf,
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
    data=fdf,
    x="embedding_x",
    y="embedding_y",
    hue="split_random",
    s=5,
    alpha=1,
)
# %% [markdown]
# Datasets are thoroughly mixed, as expected with a random split.
# This isn't good for training, we need a different split method to separate train and test datasets.

# %%
def calculate_similarities(df, split_col, fingerprint_col):
    keys = ["train", "valid", "test"]
    fingerprints = {}
    for key in keys:
        fingerprints[key] = df.loc[df[split_col] == key, fingerprint_col].to_list()
    
    key_pairs = [*((a, a) for a in keys), *((a, b) for i, a in enumerate(keys) for b in keys[i+1:])]
    similarities = {}
    for a, b in key_pairs:
        similarities[f"{a}_{b}"] = np.array([BulkTanimotoSimilarity(e, fingerprints[a]) for e in fingerprints[b]])
        # similarities[f"{b}_{a}"] = similarities[f"{a}_{b}"]
    return similarities

similarities = calculate_similarities(fdf, "split_random", column_name)
# %%
plt.hist(similarities["train_test"].max(axis=0))
# %% [markdown]
# We can see there's a lot of high maximum similarity values between train and test dataset,
# which means there's quite a bit of very similar molecules in the two datasets.
# That's bad. We need a better split.

# %%
# line histogram insted of KDE plot, because KDE is very slow for this large data set
def plot_hist(data, bins=20, range=(-5, 0), density=True, **kwargs):
    hist, bin_edges = np.histogram(np.log(data), bins=bins, range=range, density=density)
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2
    plt.plot(bin_centers, hist, **kwargs)
for key in similarities:
    plot_hist(similarities[key], label=key)
plt.legend()

# %% [markdown]
# This is even worse.
# Density plots/histograms of inter- and intra-dataset similarities are identical.
# This means the data in train, valid and test datasets is basically identically distributed and the test dataset doesn't represent any genuinely new chemistry.
# We need a better split.

# %%
sns.scatterplot(
    data=fdf,
    x="embedding_x",
    y="embedding_y",
    hue="split_scaffold",
    s=5,
    alpha=1,
)
# %% [markdown]
# With scaffold split, we see more clearly separable datasets.
# They are still mixed, but we can separate larger regions of specific colors, not only individual randomly mixed points.

#%%
similarities = calculate_similarities(fdf, "split_scaffold", column_name)
# %%
plt.hist(similarities["train_test"].max(axis=0))
# %% [markdown]
# We can see a lot less high similarity values between train and test datasets.
# Better than random split.

# %%
# line histogram insted of KDE plot, because KDE is very slow for this large data set
for key in similarities:
    plot_hist(similarities[key], label=key)
plt.legend()
# %% [markdown]
# Inter- and intra-dataset similarity score distributions are still similar, but at least they aren't identical.
# We can also see that the 3 distributions involving the train dataset are the most spread out.
# That would make some sense, because the train dataset is significantly larger (70%) than valid (10%) and test (20%).
#
# Anyway, the scaffold split is definitely better than random. Still not ideal, but better.

# %%
# choose the better split to be used in downstream code
fdf["split"] = fdf["split_scaffold"]




# %% task 5
df_5 = fdf.copy()

# %% TASK 5.1
# descributions
df_5['HeavyAtomCount'] = df_5['mol'].apply(Descriptors.HeavyAtomCount)
df_5['RingCount'] = df_5['mol'].apply(Descriptors.RingCount)
df_5['NumRotatableBonds'] = df_5['mol'].apply(Descriptors.NumRotatableBonds)
df_5['ALOGP'] = df_5['LogP']


task5_desc = [
    'MolWt',
    'HeavyAtomCount',
    'RingCount',
    'NumRotatableBonds',
    'ALOGP'
]


task5_summary_table = df_5[task5_desc].describe()

print("Task 5.1 Summary Table")
print(task5_summary_table)


# %%
task5_summary_by_split = df_5.groupby('split')[task5_desc].describe()

print("Task 5.1 Summary Table by Split")
print(task5_summary_by_split)


# %%
# Histograms 
# in the task it was written to compare across train vs test but we added also valid
for describ in task5_desc:
    plt.figure(figsize=(8, 5))
    
    sns.histplot(
        data=df_5,
        x=describ,
        hue='split',
        kde=True,
        stat='density',
        common_norm=False,
        element='step'
    )
    
    plt.title(f'Distribution of {describ} across train/test/valid')
    plt.xlabel(describ)
    plt.ylabel('Density')
    plt.tight_layout()
    plt.show()

# %% [markdown]
# The molecular weight distribution is definitely right-skewed.
# Most molecules
# are concentrated in the low molecular weight range, while a small number of 
# compounds have high molecular weight values. This indicates the presence of 
# outliers or chemically unusual large molecules. 
# The train, test, and validation subsets show rather similar distributions,
# suggesting that the split preserves the general molecular size profile of the 
# dataset.
#
#The heavy atom count distribution is also strongly right-skewed.
# Most molecules contain a relatively small number of heavy atoms,
# while a small fraction of compounds have much higher heavy atom
# counts. This pattern is consistent with the molecular weight 
# distribution.
# The train, test, and validation subsets overlap, 
# indicating that the split maintains similar size-related 
# chemical diversity across subsets.
#
#The RingCount distribution shows that most molecules contain a small number
# of rings, mainly between 0 and 5. The distribution is strongly right-skewed, 
# with only a small number of molecules containing more rings. 
# The train, test, and validation subsets overlap, although the 
# validation subset appears more concentrated around slightly higher ring 
# counts and small differences between subsets are visible.
#
#The NumRotatableBonds distribution is also strongly right-skewed. 
# Most molecules have a low number of rotatable bonds. A small fraction of
# molecules has a very high number of rotatable bonds, forming a long tail 
# toward larger values.
#The train, test, and validation distributions are similar.
#
#The ALOGP distribution is centered around low to moderate values, with most 
# molecules located approximately between 0 and 5. This suggests that the
# dataset mainly contains compounds with moderate lipophilicity. 
# The distribution also contains extreme values on both sides, 
# especially very low and very high ALOGP values, which may correspond to
#  unusual or outlier molecules.
#The train, test, and validation subsets have similar overall shapes, 
# but the validation subset appears slightly shifted toward higher ALOGP 
# values. This may indicate that in validation subset molecules are, on average, 
# slightly more lipophilic.
# %% hist log
for describ in ['MolWt', 'HeavyAtomCount']:
    plt.figure(figsize=(8, 5))
    
    sns.histplot(
        data=df_5,
        x=describ,
        hue='split',
        kde=True,
        stat='density',
        common_norm=False,
        element='step'
    )
    
    plt.xscale("log")
    plt.title(f'Log-scale distribution of {describ} across train/test/valid')
    plt.xlabel(describ)
    plt.ylabel('Density')
    plt.tight_layout()
    plt.show()

# %% [markdown]
#The log-scale MolWt distribution provides a clearer view of the main 
# molecular weight range by reducing the visual effect of extreme 
# high-molecular-weight outliers so the differences beetween subsets are more visible. 
# The train subset shows higher density at lower molecular weights, 
# while the test and validation subsets appear slightly shifted toward higher
#  molecular weights. However, the three subsets still overlap.
#
#The log-scale HeavyAtomCount distribution shows a similar pattern
# to the molecular weight distribution.
# The train subset has higher density at lower heavy atom counts, whereas the
#  test and validation subsets are slightly shifted toward higher values. 
# This indicates that test and validation may contain somewhat larger or more
#  complex molecules compared with train. This is consistent with the MolWt 
# distribution, because both descriptors describe molecular size.
# %%
# boxplots

for describ in task5_desc:
    plt.figure(figsize=(7, 5))
    
    sns.boxplot(
        data=df_5,
        x='split',
        y=describ
    )
    
    plt.yscale("log")
    plt.title(f'{describ} by split')
    plt.xlabel('Split')
    plt.ylabel(describ)
    plt.tight_layout()
    plt.show()

# %% [markdown]
#The boxplots confirm the trends observed in the histogram.
#  For size-related descriptors, such as MolWt and HeavyAtomCount, the train 
# subset has lower median values, while the test and validation subsets are 
# slightly shifted toward higher values. This suggests that test and 
# validation contain somewhat larger or more complex molecules on average.
#
#A similar pattern is visible for RingCount and ALOGP, where test and 
# especially validation tend to show slightly higher median values than 
# train. This may indicate that these subsets contain more cyclic and more 
# lipophilic compounds. For NumRotatableBonds, the differences between 
# subsets are smaller, although all subsets contain outliers with high 
# values.
#
#Overall, the boxplots show that the subsets largely overlap, but the split
#  introduces some physicochemical differences between train, test, and
#  validation. These differences may make model evaluation more realistic,
#  because the model is tested on molecules that are not perfectly identical
#  in descriptor distribution to the training set.
# %% TASK 5.2
correlation_columns = task5_desc + ['Y']

# we used sperman correlation to because it sees monotonic relations not strictly linear
correlation_matrix = df_5[correlation_columns].corr(method='spearman')

print("Task 5.2 Spearman Correlation Matrix ")
print(correlation_matrix)


# %%
# Heatmap
plt.figure(figsize=(8, 6))

sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0
)

plt.title("Spearman correlation matrix: descriptors and solubility")
plt.tight_layout()
plt.show()

# Correlations between each descriptor and the target value Y
cor_mat = (
    correlation_matrix['Y']
    .drop('Y')
    .sort_values(key=lambda x: abs(x), ascending=False)
)

print("Task 5.2 Correlations with Y")
print(cor_mat)

# %% [markdown]
#The Spearman correlation matrix shows that ALOGP has the strongest 
# correlation with the target value Y, with a correlation of -0.74. 
# This is a strong negative correlation, meaning that molecules with 
# higher ALOGP values tend to have lower aqueous solubility. This is 
# chemically meaningful, because more lipophilic molecules usually dissolve 
# less easily in water.
#
# MolWt and HeavyAtomCount also show moderate negative correlations with Y,
# equal to -0.54 and -0.51. This suggests that larger molecules tend to be 
# less soluble. 
# RingCount also has a weaker negative correlation with Y 
# (-0.40), meaning that molecules with more rings may also tend to have 
# lower solubility. 
#
# NumRotatableBonds has the weakest correlation with Y 
# (-0.17), so it appears to be the least predictive descriptor among those 
# analyzed.
#
# The features that appear most predictive are ALOGP, MolWt, and 
# HeavyAtomCount, because they have the strongest absolute correlations 
# with Y. ALOGP is especially important because it directly reflects 
# lipophilicity, which is related to water solubility.
# 
# Some correlations should be interpreted carefully. MolWt and 
# HeavyAtomCount are very strongly correlated with each other (0.93), 
# they may encode similar information about molecular size. 
# Therefore, their separate correlations with Y are not fully independent. 
# Correlations such as NumRotatableBonds with Y are small and may be less 
# useful or potentially even spurious. 


# %% TASK 6
# %% [markdown]
### Selected Features:
# - **Morgan Fingerprints**: To capture the local chemical environment of atoms (structure).
# - **LogP**: Important for solubility, as it measures lipophilicity.
# - **Molecular Weight (MolWt)**: Larger molecules often have lower solubility due to crystal lattice energy.
# - **H-Bond Donors/Acceptors**: Solubility in water is heavily driven by a molecule's ability to form hydrogen bonds with water molecules.

# %%
def embed_fingerprints(fingerprints, q=8):
    arr = np.array(list(fingerprints))
    n = arr.shape[-1]
    m = n // q
    assert m * q == n
    bit_weights = 2**np.arange(q, dtype=float)
    arr = arr.reshape((-1, m, q))
    arr = np.sum(arr * bit_weights[np.newaxis, np.newaxis, :], axis=-1)
    arr /= (2**q-1)
    return arr

# %%
embedding = embed_fingerprints(fdf.morgan_r3_l1024)

# %%
def prepare_X(df, feature_cols, fingerprint_col):
    features = df.loc[:, feature_cols].to_numpy()
    fingerprints = np.array(list(df[fingerprint_col]))
    # fingerprints = embed_fingerprints(df[fingerprint_col], 32)
    return np.hstack([features, fingerprints])
    # return features

def prepare_dataset(df):
    X = prepare_X(df, ["MolWt", "LogP", "NumHDonors"], [col for col in df.columns if col.startswith('morgan_')][0])
    y = df.Y.to_numpy()
    train_idx = df.split == "train"
    test_idx = df.split == "test"
    valid_idx = df.split == "valid"
    X_train = X[train_idx]
    y_train = y[train_idx]
    X_test = X[test_idx]
    y_test = y[test_idx]
    X_valid = X[valid_idx]
    y_valid = y[valid_idx]
    return X_train, y_train, X_test, y_test, X_valid, y_valid

X_train, y_train, X_test, y_test, X_valid, y_valid = prepare_dataset(fdf)

# %%
model = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    eval_metric=root_mean_squared_error,
    use_label_encoder=False
)

# %%
model = xgb.XGBRegressor(
    n_estimators=1000,        # large, but use early stopping
    learning_rate=0.03,      # small = safer generalization
    max_depth=6,             # shallow trees reduce overfitting
    min_child_weight=5,      # prevents tiny leaf splits
    subsample=0.7,           # row sampling
    colsample_bytree=0.3,    # VERY important for high-dim data
    reg_alpha=1.0,           # L1 regularization (feature selection)
    reg_lambda=5.0,          # L2 regularization
    gamma=1.0,               # require meaningful splits
    objective='reg:squarederror',
    tree_method='hist',      # faster + good default
    random_state=42,
    early_stopping_rounds=20,
)

# %%
model.fit(
    X_train, y_train,
    eval_set=[(X_valid, y_valid)],
    verbose=True,
)

# %%
# Predict on test set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = root_mean_squared_error(y_test, y_pred)
accuracy

# %%
def evaluate_model(X, y_true, name):
    y_pred = model.predict(X)
    rmse = root_mean_squared_error(y_true, y_pred)
    spearman_rho, _ = spearmanr(y_true, y_pred)
    print(f"{name} Set -> RMSE: {rmse}, Spearman Rho: {spearman_rho}")
    return rmse, spearman_rho

results = {
    "Train": evaluate_model(X_train, y_train, "Train"),
    "Valid": evaluate_model(X_valid, y_valid, "Valid"),
    "Test":  evaluate_model(X_test, y_test, "Test")
}
# %% [markdown]
# We established a baseline model for predicting aqueous solubility using XGBoost Regression and Morgan Fingerprints. Our final Test RMSE of 1.053 demonstrates a clear predictive trend, particularly when compared to the dataset's standard deviation of 2.37 and the inherent experimental noise in AqSolDB. While advanced architectures like Graph Neural Networks could potentially refine these results, this model effectively captures the primary variance driven by fundamental descriptors like LogP and Molecular Weight while remaining computationally efficient.

# %%
plt.scatter(y_train, model.predict(X_train), s=2, alpha=0.5, label="train")
plt.scatter(y_valid, model.predict(X_valid), s=2, alpha=0.5, label="valid")
plt.scatter(y_test, model.predict(X_test), s=2, alpha=0.5, label="test")
xrange = [np.min([np.min(y_train), np.min(y_valid), np.min(y_test)]), np.max([np.max(y_train), np.max(y_valid), np.max(y_test)])]
plt.plot(xrange, xrange, label="identity")
plt.xlabel("true")
plt.ylabel("predicted")
plt.legend()
# %% [markdown]
# We can see a strong correlation between true and predicted values in all 3 datasets.
# The deviation from identity function isn't small, but it's not too bad.
# Most points are contained inside a dense band aroud identity, its width doesn't exceed about 15% of the data range.
# We can see that the model doesn't correctly predict the lowest extremes - it seems like it limits its outputs to around -8, which contains the majority of the data points, but excludes some outliers which should go all the way to -12, but are clamped by the model to -8.
# %%