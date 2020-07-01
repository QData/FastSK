import os
import os.path as osp
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.style as style
style.use('ggplot')
plt.rcParams['axes.facecolor']='w'
plt.rcParams['axes.edgecolor']='black'
plt.rcParams['grid.color']='#abbbc6'
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, AutoMinorLocator
import matplotlib.ticker as mticker
import seaborn as sns
import numpy as np
import string
from scipy import special
import argparse


# replace '../../fastsk.xlsx' with filepath
all_data = pd.read_excel (r'fastsk.xlsx', sheet_name = 'FigureAll')
count = 0

# A new large figures with 5 per row 
data = all_data
df = pd.DataFrame(data, columns= ['Fraction', 'FastSK', 'CharCNN'])
i = 0 
titles, x, y1, y2 = [], [], [], []
while not df[i*5:i*5+5].empty:
	a = []
	subset = df[i*5:i*5+5]
	titles.append(subset['Fraction'].values.tolist()[0])
	x.append(subset['Fraction'].values.tolist()[2:])
	y1.append(subset['FastSK'].values.tolist()[2:])
	y2.append(subset['CharCNN'].values.tolist()[2:])
	i += 1

fig, axes = plt.subplots(i//5+min(1,i%5),5, figsize=(12, 10))
plt.setp(axes, xticks=np.arange(0,1.25,.25))

for n in range(i):
	axes[n//5,n%5].plot(x[n],y1[n],'bo-', label = 'FastSK')
	axes[n//5,n%5].plot(x[n],y2[n],'ro--', label = 'charCNN')
	axes[n//5,n%5].set_title(titles[n])
	#axes[n//5,n%5].set_yticks(np.arange(0.4,1.0,.2))
	handles, labels = axes[n//5,n%5].get_legend_handles_labels()

legend = fig.legend(handles, labels, prop={'size': 15}, edgecolor='black', fontsize='x-large', loc='best', bbox_to_anchor=(0.92,0.12))
fig.delaxes(axes[5,4])
fig.delaxes(axes[5,3])

# add figure axes labels
fig.text(0.5, 0.00, 'Traning Size Ratio', ha='center', fontsize=15)
fig.text(0.00, 0.5, 'Area Under the ROC Curve', va='center', rotation='vertical', fontsize=16)
fig.tight_layout()

outfile = "FigureAllFastSKvsCNNVaryTr.pdf"
print("Saving to {}".format(outfile))
plt.savefig(outfile)
plt.show()
