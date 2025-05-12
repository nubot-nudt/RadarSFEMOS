import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager
from matplotlib.pyplot import MultipleLocator


x = np.arange(0, 101, 20)
y11 = [0.1407, 0.1242, 0.1242, 0.1141, 0.1091, 0.1107] 
y12 = [0.1215, 0.1072, 0.1074, 0.1110, 0.1062, 0.1067]
y21 = [0.4991, 0.5446, 0.5533, 0.5816, 0.6081, 0.5995]
y22 = [0.5792, 0.6179, 0.6295, 0.6073, 0.6307, 0.6231]

#plt.rcParams['font.family'] = 'Times New Roman'
#plt.rcParams['font.family'] = 'serif'
#plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(x, y11, color = (168/255, 209/255, 239/255), marker = 's', markersize=8, label = 'CMFlow - EPE [m]')
ax1.plot(x, y12, color = (218/255, 178/255, 201/255), marker = 's', markersize=8, label = 'CMFlow (T) - EPE [m]')
ax2.plot(x, y21, color = (144/255, 195/255, 173/255), marker = '8', markersize=8, label = 'CMFlow - AccR')
ax2.plot(x, y22, color = (220/255, 161/255, 124/255), marker = '8', markersize=8, label = 'CMFlow (T) - AccR')

ax1.set_xlabel('Percentage of added training data [%]',fontsize = 16)
ax1.set_ylabel('EPE [m]',fontsize = 16)
ax2.set_ylabel('AccR',fontsize = 16)
ax1.set_xlim(-5,105)
y1_locator = MultipleLocator(0.01)
y2_locator = MultipleLocator(0.04)
ax1.yaxis.set_major_locator(y1_locator)
ax2.yaxis.set_major_locator(y2_locator)
ax1.tick_params(labelsize=16)
ax2.tick_params(labelsize=16)
ax1.set_ylim(0.085, 0.145)
ax1.grid(linestyle='--', linewidth=0.5)
ax2.set_ylim(0.42, 0.66)
ax2.grid(linestyle='--', linewidth=0.5)



lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc=3, ncol = 2, fontsize = 12)

fig.set_size_inches(8, 3.25)
fig.tight_layout()
#plt.savefig('add_unanno_data.png',dpi=600)
plt.savefig('add_unanno_data.pdf', format="pdf", bbox_inches="tight")