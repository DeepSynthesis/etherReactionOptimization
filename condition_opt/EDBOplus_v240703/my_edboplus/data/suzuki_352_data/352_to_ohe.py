import pandas as pd

df = pd.read_csv('dataset_B1.csv')
ohe_columns = ['ligand', 'base', 'solvent']
df_last = df[ohe_columns].copy()
df_sampling = pd.get_dummies(df, prefix=ohe_columns,
                                columns=ohe_columns, drop_first=True)
for col in ohe_columns :
    df_sampling[col] = df_last[col]
df_sampling.drop(columns=['ligand_equivalent'], inplace=True)
df_sampling.to_csv('dataset_B1_OHE_clean.csv')

df_read = pd.read_csv('dataset_B1_OHE_clean.csv')
print(df_read.columns)