import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import MultipleLocator

# for direct thresholding
xx = np.arange(0.4, 5.2, 0.4)
yy = [0.274, 0.3723,0.4483,	0.4676,	0.4754,	0.4782,	0.4795,	0.484,	0.4835,	0.4788,	0.4745,	0.4684]
yy2 = [0.4701, 0.641, 0.7672, 0.8068, 0.8294, 0.8438, 0.8551, 0.869,	0.8769,	0.8786,	0.8788,	0.8789]
#plt.rcParams['font.family'] = 'serif'
#plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
fig = plt.figure(figsize=(10, 3))
plt.plot(xx, yy, color='#FF81C0', marker='o', markersize=8)
plt.grid(linestyle='--', linewidth=0.5)
#plt.scatter([3.2], [0.4840], c='r')
plt.ylim(0.25,0.5)
plt.xlim(0.2, 5.0)
plt.legend(['direct thresholding'], fontsize=16, loc = 0)
plt.xlabel('threshold [m/s]', size=16)
plt.ylabel('mIoU', size=16)
x_locator = MultipleLocator(0.4)
y_locator = MultipleLocator(0.05)
ax = plt.gca()
ax.xaxis.set_major_locator(x_locator)
ax.yaxis.set_major_locator(y_locator)
plt.tick_params(labelsize=16)
fig.tight_layout()
plt.savefig('miou_thres_direct.pdf', format="pdf", bbox_inches="tight")