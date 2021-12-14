import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


task = 'GoNogo' #'PerceptualDecisionMaking' # 
df = []
for subj in range(70):
    df_= pd.read_csv(f'/Users/laurasuarez/Desktop/{task}_{subj}.csv', index_col=False,
                     usecols=['module','alpha','score'])
    df_['subj_id'] = subj
    df.append(df_[['subj_id','module','alpha','score']])

df = pd.concat(df, ignore_index=False).reset_index()
df['score'] = df['score'].astype(float)

# print(np.min(df.score))
# print(np.max(df.score))
# df['score_'] = (df['score']-np.min(df['score']))/(np.max(df['score'])-np.min(df['score']))
# print(np.min(df.score_))
# print(np.max(df.score_))

sns.set(style="ticks", font_scale=2.0)  
fig = plt.figure(num=1, figsize=(12,10))
ax = plt.subplot(111)
# ax = plt.figure(num=1, figsize=(12,5), constrained_layout=True).subplots(1, 1)
sns.lineplot(data=df, x='alpha', y='score', 
             hue='module', 
             hue_order=['VIS', 'SM', 'DA', 'VA', 'LIM', 'FP', 'DMN'],
             palette=sns.color_palette('husl', 7), 
             markers=True, 
             ax=ax)
# ax.set_ylim(0.0,1.0)
sns.despine(offset=10, trim=True)
plt.title(task)
plt.plot()
plt.show()