import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1
df = pd.read_csv('medical_examination.csv')

# 2
df['overweight'] = (df['weight'] / (df['height'] / 100) ** 2).apply(lambda x: 1 if x > 25 else 0)

# 3
df['cholesterol'] = df['cholesterol'].apply(lambda x: 0 if x == 1 else 1)
df['gluc'] = df['gluc'].apply(lambda x: 0 if x == 1 else 1)

# 4
def draw_cat_plot():
    # 5: Melt the data for categorical plot
    df_cat = pd.melt(df, id_vars=['cardio'], value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])

    # 6: Group and format data by cardio, variable, and value counts
    df_cat = df_cat.groupby(['cardio', 'variable', 'value'], as_index=False).size()

    # Rename 'size' column for catplot
    df_cat = df_cat.rename(columns={'size': 'total'})

    # 7: Create the categorical plot using sns.catplot
    fig = sns.catplot(x='variable', y='total', hue='value', col='cardio', data=df_cat, kind='bar').fig

    # 8: Save and return the figure
    fig.savefig('catplot.png')
    return fig

# 10
def draw_heat_map():
    # 11: Clean the data
    df_heat = df[(df['ap_lo'] <= df['ap_hi']) & 
                 (df['height'] >= df['height'].quantile(0.025)) & 
                 (df['height'] <= df['height'].quantile(0.975)) & 
                 (df['weight'] >= df['weight'].quantile(0.025)) & 
                 (df['weight'] <= df['weight'].quantile(0.975))]

    # 12: Calculate the correlation matrix
    corr = df_heat.corr()

    # 13: Generate mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 14: Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # 15: Plot the heatmap
    sns.heatmap(corr, mask=mask, annot=True, fmt=".1f", linewidths=.5, ax=ax)

    # 16: Save and return the figure
    fig.savefig('heatmap.png')
    return fig
