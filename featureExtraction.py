import numpy as np
import pywt


def get_attention_score(features):
    '''
    获得当前帧的瞬时专注度
    :param oneframe: 一帧脑电数据
    :param fs: 信号采样率
    :return: 当前专注度得分
    '''
    # frameLength = len(oneframe)
    # y_theta = sum(abs(rhythmExtraction(
    #     oneframe, 4, 7, fs, frameLength, 'θ wave'))) / frameLength
    # y_alpha1 = sum(abs(rhythmExtraction(
    #     oneframe, 8, 10, fs, frameLength, 'α_low wave'))) / frameLength
    # y_alpha2 = sum(abs(rhythmExtraction(
    #     oneframe, 11, 13, fs, frameLength, 'α_high wave'))) / frameLength
    # y_beta1 = sum(abs(rhythmExtraction(
    #     oneframe, 14, 20, fs, frameLength, 'β_low wave'))) / frameLength
    # y_beta2 = sum(abs(rhythmExtraction(
    #     oneframe, 21, 30, fs, frameLength, 'β_high wave'))) / frameLength
    attention_score = (0.6 * features["beta"]) / (features["alpha"] + features["theta"] + features['delta'])
    return attention_score


# 需要分析的四个频段
iter_freqs = [
    {'name': 'Delta', 'fmin': 0, 'fmax': 4},
    {'name': 'Theta', 'fmin': 4, 'fmax': 8},
    {'name': 'Alpha', 'fmin': 8, 'fmax': 13},
    {'name': 'Beta', 'fmin': 13, 'fmax': 35},
]


def get_rhythm_features(data, fs, wavelet, maxlevel=8):
    '''
    提取信号节律波特征值 # 小波包分解
    :param oneframe: eeg信号
    :param fs: 采样频率
    :return: 特征值集合
    '''
    wp = pywt.WaveletPacket(data=data, wavelet=wavelet, mode='symmetric', maxlevel=maxlevel)
    # 频谱由低到高的对应关系，这里需要注意小波变换的频带排列默认并不是顺序排列，所以这里需要使用’freq‘排序。
    freqTree = [node.path for node in wp.get_level(maxlevel, 'freq')]
    # 计算maxlevel最小频段的带宽
    freqBand = fs / (2 ** maxlevel)
    # 定义能量数组
    energy = []
    # 循环遍历计算四个频段对应的能量
    for iter in range(len(iter_freqs)):
        iterEnergy = 0.0
        for i in range(len(freqTree)):
            # 第i个频段的最小频率
            bandMin = i * freqBand
            # 第i个频段的最大频率
            bandMax = bandMin + freqBand
            # 判断第i个频段是否在要分析的范围内
            if (iter_freqs[iter]['fmin'] <= bandMin and iter_freqs[iter]['fmax'] >= bandMax):
                # 计算对应频段的累加和
                iterEnergy += pow(np.linalg.norm(wp[freqTree[i]].data, ord=None), 2)
        # 保存四个频段对应的能量和
        energy.append(iterEnergy)
    # # 绘制能量分布图
    # plt.plot([xLabel['name'] for xLabel in iter_freqs], energy, lw=0, marker='o')
    # plt.title('能量分布')
    # plt.show()
    spectral_feature = {"delta": energy[0], "theta": energy[1], "alpha": energy[2], "beta": energy[3]}
    return spectral_feature


################  移植看这里 ####################
def get_rhythm_features_fft(data, fs):
    spectral_feature = {"delta": 0, "theta": 0, "alpha": 0, "beta": 0}
    data_fft = abs(np.fft.fft(data, 512))
    N = len(data_fft)
    data_fft = data_fft[0:int(N / 2)]
    fr = np.linspace(0, fs, int(N / 2))
    t = np.arange(0, len(data) / fs, 1.0 / fs)
    for i, item in enumerate(fr):
        if 0 < item < 4:
            spectral_feature["delta"] += data_fft[i] ** 2
        elif 4 < item < 8:
            spectral_feature["theta"] += data_fft[i] ** 2
        elif 8 < item < 13:
            spectral_feature["alpha"] += data_fft[i] ** 2
        elif 13 < item < 35:
            spectral_feature["beta"] += data_fft[i] ** 2
    return spectral_feature


################  END ####################

def get_meditation_score(features):
    '''
    获得当前帧的瞬时放松度
    :param oneframe: 一帧脑电数据
    :param fs: 信号采样率
    :return: 当前放松度得分
    '''
    meditation_score = (features["alpha"]) / (features["alpha"] + features["theta"] + features["beta"])
    return meditation_score


def rhythmExtraction(oneFrame, f_low, f_high, fs, frameLength, title):
    data_fft = np.fft.fft(oneFrame)
    f1 = round(frameLength / fs * f_low + 1)
    f2 = round(frameLength / fs * f_high + 1)
    f3 = round(frameLength / fs * (fs - f_high) + 1)
    f4 = round(frameLength / fs * (fs - f_low) + 1)

    data_fft[1: f1] = 0
    data_fft[f2: f3] = 0
    data_fft[f4: frameLength] = 0
    y_time = np.fft.ifft(data_fft)
    # wavplot(y_time,title)
    return y_time


def get_order_diff_quot(xi=[], fi=[]):
    if len(xi) > 2 and len(fi) > 2:
        return (get_order_diff_quot(xi[:len(xi) - 1], fi[:len(fi) - 1]) - get_order_diff_quot(xi[1:len(xi)],
                                                                                              fi[1:len(fi)])) / float(
            xi[0] - xi[-1])
    return (fi[0] - fi[1]) / float(xi[0] - xi[1])


def get_Wi(i=0, xi=[]):
    def Wi(x):
        result = 1.0
        for each in range(i):
            result *= (x - xi[each])
        return result
    return Wi


def get_Newton_inter(xi=[], fi=[]):
    def Newton_inter(x):
        result = fi[0]
        for i in range(2, len(xi)):
            result += (get_order_diff_quot(xi[:i], fi[:i]) * get_Wi(i - 1, xi)(x))
        return result
    return Newton_inter


def data_interpolation(data, enlarge):
    total_len = len(data) + enlarge
    x = np.linspace(0, len(data), len(data))
    x_new = np.linspace(0, len(data), total_len)
    Nx = get_Newton_inter(x, data)
    ynew = Nx(x_new)
    return ynew


if __name__ == '__main__':
    # 插值法验证
    import math
    data = [math.sin(item) for item in range(10)]
    enlarge_num = 10
    print(data)
    ynew = data_interpolation(data, enlarge_num)
    import pylab as pl
    pl.subplot(211)
    pl.plot(data, marker='*')
    pl.subplot(212)
    pl.plot(ynew, marker='.')
    pl.savefig('chazhi.png')
    pl.show()
