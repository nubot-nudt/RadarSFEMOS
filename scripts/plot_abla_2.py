import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import MultipleLocator

# for bias compensated thresholding 
xx = np.arange(0.1, 1.3, 0.1)
yy = [0.5088,0.5182,0.5203,	0.5186,	0.5171,	0.5163,	0.5143,	0.5122,	0.5105,	0.5084,	0.5067,	0.5055]
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
fig = plt.figure(figsize=(10, 3))
plt.plot(xx, yy,'c-o', markersize=8)
plt.grid(linestyle='--', linewidth=0.5)
#plt.scatter([3.2], [0.4840], c='r')
plt.ylim(0.5, 0.525)
plt.xlim(0.05, 1.25)
plt.legend(['bias-aware thresholding'], fontsize=16, loc = 3)
plt.xlabel('threshold [m/s]', size=16)
plt.ylabel('mIoU', size=16)
x_locator = MultipleLocator(0.1)
y_locator = MultipleLocator(0.005)
ax = plt.gca()
ax.xaxis.set_major_locator(x_locator)
ax.yaxis.set_major_locator(y_locator)
plt.tick_params(labelsize=16)
fig.tight_layout()
plt.savefig('miou_thres_bias.pdf', format="pdf", bbox_inches="tight")