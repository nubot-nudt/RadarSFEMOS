import matplotlib.pyplot as plt


label_list = ['$T_r=1$', '$T_r=3$', '$T_r=5$', '$T_r=10$',  '$T_r=20$']    
num_list1 = [0.1376, 0.1332, 0.1297, 0.1297, 0.1343]      
num_list2 = [0.1933, 0.2117, 0.2277, 0.2316, 0.2211]      
x = range(len(num_list1))
"""
绘制条形图
left:长条形中点横坐标
height:长条形高度
width:长条形宽度，默认值0.8
label:为后面设置legend准备
"""
rects1 = plt.bar(x=x, height=num_list1, width=0.4, alpha=0.8, color='red', label='EPE [m]')
rects2 = plt.bar(x=[i + 0.4 for i in x], height=num_list2, width=0.4, color='green', label="AccS")
plt.ylim(0, 0.5)     
"""
设置x轴刻度显示值
参数一：中点坐标
参数二：显示值
"""
plt.xticks([index + 0.2 for index in x], label_list, fontfamily = 'monospace', size = 10)
plt.legend()     

for rect in rects1:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width() / 2, height+1, str(height), ha="center", va="bottom")
for rect in rects2:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width() / 2, height+1, str(height), ha="center", va="bottom")
plt.savefig('T_r_inference.png', dpi=400)