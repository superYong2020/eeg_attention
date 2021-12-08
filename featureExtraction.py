import numpy as np


def get_attention_score(features):
    '''
    获得当前帧的瞬时专注度
    :param oneframe: 一帧脑电数据
    :param fs: 信号采样率
    :return: 当前专注度得分
    '''
    weight = [2, 1, 1]
    # attn_score = w_1 * avg_beta / w_2 * avg_alpha + w_3 * avg_theta
    print(features)
    attention_score = (weight[0] * (features["low_beta"] + features['high_beta']) / 2) / (
            weight[1] * (features["low_alpha"] + features['high_alpha']) / 2 + weight[2] * features["theta"])
    return attention_score


def get_meditation_score(features):
    '''
    获得当前帧的瞬时放松度
    :param oneframe: 一帧脑电数据
    :param fs: 信号采样率
    :return: 当前放松度得分
    '''
    # medit_score = avg_alpha / avg_alpha + avg_theta + avg_beta
    meditation_score = ((features["low_alpha"] + features['high_alpha']) / 2) / (
            (features["low_alpha"] + features['high_alpha']) / 2 + features["theta"] + (
            features["low_beta"] + features['high_beta']) / 2)
    return meditation_score


# 需要分析的四个频段
iter_freqs = {
    'delta': {'fmin': 0, 'fmax': 4},
    'theta': {'fmin': 4, 'fmax': 8},
    'low_alpha': {'fmin': 8, 'fmax': 10},
    'high_alpha': {'fmin': 10, 'fmax': 13},
    'low_beta': {'fmin': 13, 'fmax': 20},
    'high_beta': {'fmin': 20, 'fmax': 35},
    'low_gamma': {'fmin': 35, 'fmax': 50},
    'high_gamma': {'fmin': 50, 'fmax': 100}
}


def get_rhythm_features_fft(data, fs):
    spectral_feature = {item: [] for item in iter_freqs.keys()}
    data_fft = abs(np.fft.fft(data, 128))  # 更改fft点数
    N = len(data_fft)
    data_fft = data_fft[0:int(N / 2)]
    fr = np.linspace(0, 128, int(N / 2))  # 更改f映射
    t = np.arange(0, len(data) / fs, 1.0 / fs)
    for i, item in enumerate(fr):
        if iter_freqs['delta']['fmin'] < item < iter_freqs['delta']['fmax']:
            spectral_feature["delta"].append(data_fft[i] ** 2)
        elif iter_freqs['theta']['fmin'] < item < iter_freqs['theta']['fmax']:
            spectral_feature["theta"].append(data_fft[i] ** 2)
        elif iter_freqs['low_alpha']['fmin'] < item < iter_freqs['low_alpha']['fmax']:
            spectral_feature["low_alpha"].append(data_fft[i] ** 2)
        elif iter_freqs['high_alpha']['fmin'] < item < iter_freqs['high_alpha']['fmax']:
            spectral_feature["high_alpha"].append(data_fft[i] ** 2)
        elif iter_freqs['low_beta']['fmin'] < item < iter_freqs['low_beta']['fmax']:
            spectral_feature["low_beta"].append(data_fft[i] ** 2)
        elif iter_freqs['high_beta']['fmin'] < item < iter_freqs['high_beta']['fmax']:
            spectral_feature["high_beta"].append(data_fft[i] ** 2)
        elif iter_freqs['low_gamma']['fmin'] < item < iter_freqs['low_gamma']['fmax']:
            spectral_feature["low_gamma"].append(data_fft[i] ** 2)
        elif iter_freqs['high_gamma']['fmin'] < item < iter_freqs['high_gamma']['fmax']:
            spectral_feature["high_gamma"].append(data_fft[i] ** 2)
    out = {}
    for key, value in spectral_feature.items():
        out[key] = np.mean(value)
    return out


################  END ####################


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


def data_interpolation_old(data, enlarge):
    total_len = len(data) + enlarge
    x = list(np.linspace(0, len(data) - 1, len(data)))
    x_new = x.copy()
    # x_new.extend([x[-1] + 0.2])
    print(x)
    print(x_new)
    Nx = get_Newton_inter(x, data)
    ynew = Nx(x_new)
    return ynew


def data_interpolation(data, enlarge):
    low = np.min(data[-2:])
    high = np.max(data[-2:])
    if low == high:
        low = 0 if low < 1 else low - 1
    out = np.random.randint(low=low, high=high, size=enlarge)
    return out
