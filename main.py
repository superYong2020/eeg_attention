# -*- coding: utf-8 -*-
# @Time    : 2021/8/31 13:03 
# @Author  : Yong Cao
# @Email   : yongcao_epic@hust.edu.cn
import json
from tkinter import _flatten
import numpy as np
from featureExtraction import get_attention_score, get_meditation_score, get_rhythm_features_fft

if __name__ == '__main__':
    ## 读取data, 定义采样频率
    raw_path = "./data/eegRaw.json"
    data = []
    sfreq = 512.0
    with open(raw_path, 'r') as f:
        json_str = json.load(f)
        for item in json_str:
            if "value" in item.keys():
                if len(item['value']) > 200:
                    data.append(item['value'])
        result = list(_flatten(data))
    ## 数据分帧 windows = 128 即128ms计算一次专注度
    window = 128
    observe_window = 3000  #观测过去3秒的平均专注度
    attention_cache = []
    meditation_cache = []

    for i in range(100):
        eegData = np.array(result[window * i:window * (i + 1)])
        # fft获取节律波特征
        spectral_feature = get_rhythm_features_fft(eegData, sfreq)
        print("fft rhythm_features is \n", spectral_feature.values())
        # 计算attention得分
        atten_score = get_attention_score(spectral_feature)
