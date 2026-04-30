import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os


df = pd.read_csv('../data/migrainedata.csv')
print(df['Type'].value_counts())
sns.countplot(x='Type', data=df)
plt.xticks(rotation=45)
plt.show()