import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 示例：仅绘制 Rainy 场景的数据
methods = ['w/o RGB', 'w/o IR', 'w/o AIS', 'w/o Tensor', 'w/o TransNet', 'TAOTNet']
accuracy = [73.08, 38.46, 88.46, 80.77, 84.62, 92.31]
precision = [68.69, 22.91, 93.39, 76.92, 92.59, 95.37]
recall = [65.00, 38.33, 90.00, 83.33, 81.67, 93.33]
f1 = [63.19, 28.57, 89.10, 79.37, 82.54, 93.34]

x = np.arange(len(methods))
width = 0.2

fig, ax = plt.subplots(figsize=(10, 6))
# 绘制四组柱子
rects1 = ax.bar(x - 1.5*width, accuracy, width, label='Accuracy', color='#A9C4EB')
rects2 = ax.bar(x - 0.5*width, precision, width, label='Precision', color='#84A59D')
rects3 = ax.bar(x + 0.5*width, recall, width, label='Recall', color='#F6BD60')
rects4 = ax.bar(x + 1.5*width, f1, width, label='F1', color='#F28482')

# 突出 TAOTNet 的效果 (可选)
# 比如给最后一个 Method 的柱子加粗边框或换成高饱和度颜色

ax.set_ylabel('Score (%)')
ax.set_title('Rainy Condition')
ax.set_xticks(x)
ax.set_xticklabels(methods, rotation=15)
ax.legend()

plt.tight_layout()
plt.savefig('rainy_ablation.pdf', format='pdf', dpi=300) # 导出为PDF供LaTeX使用
plt.show()