# -*- coding: utf-8 -*-
# @Time    : 2021/8/31 13:03 
# @Author  : Yong Cao
# @Email   : yongcao_epic@hust.edu.cn
import json
from tkinter import _flatten
import numpy as np
from featureExtraction import HHTFilter, get_attention_score, smooth_atten_score

if __name__ == '__main__':
    ## 读取data, 定义采样频率
    raw_path = "./data/eegRaw.json"
    data = []
    sfreq = 512
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
    for i in range(100):
        eegData = np.array(result[window * i:window * (i + 1)])
        eegRetain = HHTFilter(eegData, [0, 1])
        # 计算attention得分
        atten_score = get_attention_score(eegData, sfreq)
        attention_cache.append(atten_score)
        # 均值窗口输出
        out_score = smooth_atten_score(attention_cache, observe_window, window)
        print("windows {} attention score is {:.0f}.".format(i, out_score * 100))