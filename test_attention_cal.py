# -*- coding: utf-8 -*-
# @Time    : 2021/10/29 20:58 
# @Author  : Yong Cao
# @Email   : yongcao_epic@hust.edu.cn
import matplotlib.pyplot as plt
from featureExtraction import get_attention_score, get_rhythm_features_fft

with open('data_new/3_raw.txt', 'r') as f:
    data = [int(item) for item in f.read().split('\n') if item != '' and item != '-']

with open('data_new/3_att.txt', 'r') as f:
    data_attn = [int(item) for item in f.read().split('\n') if item != '' and item != '-']

frame = 0
sfreq = 512
windows = 512
atten_score_our = []
alpha = []
theta = []
beta = []
while (frame + 1) * windows < len(data):
    eeg_frame = data[frame:frame + windows]
    spectral_feature = get_rhythm_features_fft(eeg_frame, sfreq)
    atten_score = get_attention_score(spectral_feature)
    atten_score_our.append(atten_score * 100)
    frame += 1

x = [item / sfreq for item in range(len(data))]


## 可视化，skip
plt.subplot(311)
plt.title("experi 3: 5-10 专注 30-40 不专注 50-55 专注 60-65 不专注")
plt.plot(x, data)
line_list = [5,10,30,40,50,55,60,65]
for item in line_list:
    plt.axvline(512 * item / sfreq, color='red')
plt.subplot(312)
plt.plot(data_attn, label="大包")
for item in line_list:
    plt.axvline(item, color='red')
plt.legend(loc='lower right', fontsize=11, frameon=True, fancybox=True, framealpha=0.8,
           borderpad=0.3, ncol=1,
           markerfirst=True, markerscale=1, numpoints=1, handlelength=0.2)
plt.subplot(313)
plt.plot(atten_score_our, label="小包")
plt.legend(loc='lower right', fontsize=11, frameon=True, fancybox=True, framealpha=0.8,
           borderpad=0.3, ncol=1,
           markerfirst=True, markerscale=1, numpoints=1, handlelength=0.2)
for item in line_list:
    plt.axvline(item, color='red')

plt.show()
