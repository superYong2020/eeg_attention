# -*- coding: utf-8 -*-
# @Time    : 2021/10/29 20:58 
# @Author  : Yong Cao
# @Email   : yongcao_epic@hust.edu.cn
import matplotlib.pyplot as plt

from featureExtraction import get_attention_score, get_rhythm_features_fft

with open('data_new/raw3.txt', 'r') as f:
    data = [int(item) for item in f.read().split('\n') if item != '' and item != '-']

with open('data_new/att3.txt', 'r') as f:
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
    alpha.append(spectral_feature['alpha'])
    theta.append(spectral_feature['theta'])
    beta.append(spectral_feature['beta'])

x = [item / sfreq for item in range(len(data))]
from matplotlib.font_manager import FontProperties

font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
plt.subplot(311)
# plt.title("实验2：5-10 专注 / 25-30 不专注 / 40-45 专注 / 55-60 不专注", fontproperties=font)
# plt.title("6-12 专 / 12-20 不 / 20-32 专 / 32-40 不 / 40-50 专 / 50-62 不", fontproperties=font)
plt.plot(x, data)
# line_list = [5,10,25,30,40,45,55,60]
# line_list = [8,20,22,36,54,74,80,96,100,111,115,122,124,134,155,171,180]
line_list = [3,11,13,25,55,82,100,120,130,144,156,167]
for item in line_list:
    plt.axvline(512 * item / sfreq, color='red')
plt.subplot(312)
plt.plot(data_attn, label="大包")
for item in line_list:
    plt.axvline(item, color='red')
plt.legend(loc='lower right', fontsize=11, frameon=True, fancybox=True, framealpha=0.8,
           borderpad=0.3, ncol=1, prop=font,
           markerfirst=True, markerscale=1, numpoints=1, handlelength=0.2)
plt.subplot(313)
plt.plot(atten_score_our, label="小包")
plt.legend(loc='lower right', fontsize=11, frameon=True, fancybox=True, framealpha=0.8,
           borderpad=0.3, ncol=1, prop=font,
           markerfirst=True, markerscale=1, numpoints=1, handlelength=0.2)
for item in line_list:
    plt.axvline(item, color='red')

# plt.subplot(614)
# plt.plot(alpha, label="alpha")
# plt.legend(loc='lower right', fontsize=11, frameon=True, fancybox=True, framealpha=0.8,
#            borderpad=0.3, ncol=1, prop=font,
#            markerfirst=True, markerscale=1, numpoints=1, handlelength=0.2)
#
# plt.subplot(615)
# plt.plot(beta, label="beta")
# plt.legend(loc='lower right', fontsize=11, frameon=True, fancybox=True, framealpha=0.8,
#            borderpad=0.3, ncol=1, prop=font,
#            markerfirst=True, markerscale=1, numpoints=1, handlelength=0.2)
#
# plt.subplot(616)
# plt.plot(theta, label="theta")
# plt.legend(loc='lower right', fontsize=11, frameon=True, fancybox=True, framealpha=0.8,
#            borderpad=0.3, ncol=1, prop=font,
#            markerfirst=True, markerscale=1, numpoints=1, handlelength=0.2)


plt.savefig('exper_3.png')
plt.show()
