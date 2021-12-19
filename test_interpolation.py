# -*- coding: utf-8 -*-
# @Time    : 2021/11/21 3:48 下午 
# @Author  : Yong Cao
# @Email   : yongcao_epic@hust.edu.cn
from featureExtraction import data_interpolation
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # 预先读取10个数据
    data_queue = [0] * 10
    plot_queue = []
    windows = 10
    with open('data_new/1_att.txt', 'r') as f:
        data_attn = [int(item) for item in f.read().split('\n') if item != '' and item != '-']

    origin = []
    for t in range(100):
        if windows + t >= len(data_attn):
            break
        data_queue.pop(0)
        data_queue.append(data_attn[windows + t])
        # 根据前十个数预测后面的n个数据
        out = data_interpolation(data_queue, 4)
        plot_queue.append(data_attn[windows + t])
        plot_queue.extend(out)
        origin.append(data_attn[windows + t])

    plt.subplot(211)
    plt.plot(origin, label="before")
    plt.title("Attention score")
    plt.legend(loc='lower right', fontsize=11, frameon=True, fancybox=True, framealpha=0.8,
               borderpad=0.3, ncol=1,
               markerfirst=True, markerscale=1, numpoints=1, handlelength=0.2)
    plt.subplot(212)
    plt.plot(plot_queue, label="after")
    plt.legend(loc='lower right', fontsize=11, frameon=True, fancybox=True, framealpha=0.8,
               borderpad=0.3, ncol=1,
               markerfirst=True, markerscale=1, numpoints=1, handlelength=0.2)
    plt.show()