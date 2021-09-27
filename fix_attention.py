# -*- coding: utf-8 -*-
# @Time    : 2021/9/27 11:26 
# @Author  : Yong Cao
# @Email   : yongcao_epic@hust.edu.cn
import os
import matplotlib.pyplot as plt
import numpy as np
from featureExtraction import get_attention_score, get_rhythm_features, get_meditation_score

# 基础配置
eeg = []
attn = []
windows = 650
sfreq = 512
atten_score_our = []

# 读数据
with open('./Data/rawAndAtt.txt', 'r') as f:
    for row in f.readlines():
        data = [item.strip() for item in row.strip().split(":")]
        if "Raw" in data[1]:
            eeg.append(int(data[2]))
        else:
            attn.append(int(data[2]))

# 分窗 求每个窗口的专注度
index = 0
while (index+1)*windows < len(eeg):
    check_window = np.array(eeg[index*windows:(index+1)*windows])
    valid = np.where(abs(check_window) > 800)[0].shape[0] < 10
    spectral_feature = get_rhythm_features(check_window, sfreq, 'db4')
    if valid:
        atten_score = get_attention_score(spectral_feature)
    else:
        atten_score = 0
    if atten_score > 1:
        atten_score = 1
    atten_score_our.append(atten_score*100)
    index += 1

# 三帧窗口平均平滑
new_y = []
for i in range(len(atten_score_our)):
    if i < 3:
        new_y.append(atten_score_our[i])
    else:
        new_y.append(np.mean(atten_score_our[i-3:i]))


## 可视化
from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
plt.subplot(211)
font_size=12
plt.plot(attn, color="#ff585d", label='神念')
plt.yticks(fontsize=font_size)
plt.title("算法对比效果", fontproperties=font)
plt.ylabel(ylabel='专注度值', fontsize=font_size, fontproperties=font)
plt.legend(loc='upper left', fontsize=font_size, frameon=True, fancybox=True, framealpha=1,
           borderpad=0.3, ncol=1, prop=font,
           markerfirst=True, markerscale=1, numpoints=1, handlelength=3.5)
plt.axis([-2, 115, -2, 105])

plt.subplot(212)
plt.plot(new_y, color="#41b6e6", label="我们的")
plt.xticks(fontsize=font_size)
plt.yticks(fontsize=font_size)
plt.xlabel(xlabel='时间', fontsize=font_size, fontproperties=font)
plt.ylabel(ylabel='专注度值', fontsize=font_size, fontproperties=font)
plt.legend(loc='upper left', fontsize=font_size, frameon=True, fancybox=True, framealpha=1,
           borderpad=0.3, ncol=1, prop=font,
           markerfirst=True, markerscale=1, numpoints=1, handlelength=3.5)
plt.axis([-2, 115, -2, 105])
plt.savefig('./images/attention_result.png')
plt.show()
