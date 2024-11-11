import pandas as pd
import os

MAXSIMS = (1+int(os.environ['MAX_SIM']))
# Use lower quarter to estimate mislabels
flipThreshold=int(MAXSIMS/4)
# Use middle half (50%) to estimate OOD samples
ignoreThreshold=int(MAXSIMS*3/4)


# Find data to flip on training
df = pd.read_csv(os.environ['TRAINRES'],index_col=False)
predCols = [i for i,n in enumerate(df.columns) if n.startswith('Pred')]
df.loc[df['Truth']==1,'Correct'] = df[df['Truth'] == 1].iloc[:,predCols].sum(1)
df.loc[df['Truth']==0,'Correct'] = MAXSIMS - df[df['Truth'] == 0].iloc[:,predCols].sum(1)
toFlip = pd.DataFrame()
toFlip['Index'] = df[df['Correct'] <= flipThreshold]['Index']
toFlip.to_csv('questionableTraining.csv',index=False)
# Attempt to find OOD data through RANSAC
toIgnore = pd.DataFrame()
toIgnore['Index'] = df.loc[(df['Correct'] > flipThreshold) & (df['Correct'] <= ignoreThreshold)]['Index']
toIgnore.to_csv('oodTraining.csv',index=False)


# Find data to flip on testing
df = pd.read_csv(os.environ['TESTRES'],index_col=False)
predCols = [i for i,n in enumerate(df.columns) if n.startswith('Pred')]
df.loc[df['Truth']==1,'Correct'] = df[df['Truth'] == 1].iloc[:,predCols].sum(1)
df.loc[df['Truth']==0,'Correct'] = MAXSIMS - df[df['Truth'] == 0].iloc[:,predCols].sum(1)
toFlip = pd.DataFrame()
toFlip['Index'] = df[df['Correct'] <= flipThreshold]['Index']
toFlip.to_csv('questionableTesting.csv',index=False)
# Attempt to find OOD data through RANSAC
toIgnore = pd.DataFrame()
toIgnore['Index'] = df.loc[(df['Correct'] > flipThreshold) & (df['Correct'] <= ignoreThreshold)]['Index']
toIgnore.to_csv('oodTesting.csv',index=False)


