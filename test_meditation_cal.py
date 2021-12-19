# -*- coding: utf-8 -*-
# @Time    : 2021/12/19 15:59 
# @Author  : Yong Cao
# @Email   : yongcao_epic@hust.edu.cn
import matplotlib.pyplot as plt
from featureExtraction import get_attention_score, get_rhythm_features_fft

with open('data_new/3_raw.txt', 'r') as f:
    data = [int(item) for item in f.read().split('\n') if item != '' and item != '-']

frame = 0
sfreq = 512
windows = 512
mediate_score_all = []
while (frame + 1) * windows < len(data):
    eeg_frame = data[frame:frame + windows]
    spectral_feature = get_rhythm_features_fft(eeg_frame, sfreq)
    mediate_score = get_attention_score(spectral_feature)
    mediate_score_all.append(mediate_score * 100)
    frame += 1

plt.plot(mediate_score_all)
plt.show()