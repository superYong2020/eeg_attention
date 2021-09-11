import matplotlib.pyplot as plt
import numpy as np
from pyhht import EMD


def HHTFilter(eegRaw, componentsRetain):
    '''
    滤波，去除眼电干扰
    :param eegRaw: 分帧后的数据
    :param componentsRetain: 需要保留的信号成分
    :return: 去除眼电后的干净数据
    '''
    # 进行EMD分解
    decomposer = EMD(eegRaw)
    # 获取EMD分解后的IMF成分
    imfs = decomposer.decompose()
    # 选取需要保留的EMD组分，并且将其合成信号
    eegRetain = np.sum(imfs[componentsRetain], axis=0)
    # 可视化
    # # 绘图
    # plt.figure(figsize=(10, 7))
    # # 绘制原始数据
    # plt.plot(eegRaw, label='RawData')
    # # 绘制保留组分合成的数据
    # plt.plot(eegRetain, label='HHTData')
    # # 绘制标题
    # plt.title('RawData-----HHTData',fontsize=20)
    # # 绘制图例
    # plt.legend()
    # plt.show()
    return eegRetain


# 定义HHT的计算分析函数
def HHTAnalysis(eegRaw, fs):
    # 进行EMD分解
    decomposer = EMD(eegRaw)
    # 获取EMD分解后的IMF成分
    imfs = decomposer.decompose()
    # 分解后的组分数
    n_components = imfs.shape[0]
    # # 定义绘图，包括原始数据以及各组分数据
    # fig, axes = plt.subplots(n_components + 1, 2, figsize=(10, 7), sharex=True, sharey=False)
    # # 绘制原始数据
    # axes[0][0].plot(eegRaw)
    # # 原始数据的Hilbert变换
    # eegRawHT = hilbert(eegRaw)
    # # 绘制原始数据Hilbert变换的结果
    # axes[0][0].plot(abs(eegRawHT))
    # # 设置绘图标题
    # axes[0][0].set_title('Raw Data')
    # # 计算Hilbert变换后的瞬时频率
    # instf, timestamps = tftb.processing.inst_freq(eegRawHT)
    # # 绘制瞬时频率，这里乘以fs是正则化频率到真实频率的转换
    # axes[0][1].plot(timestamps, instf * fs)
    # # 计算瞬时频率的均值和中位数
    # axes[0][1].set_title('Freq_Mean{:.2f}----Freq_Median{:.2f}'.format(np.mean(instf * fs), np.median(instf * fs)))
    #
    # # 计算并绘制各个组分
    # for iter in range(n_components):
    #     # 绘制分解后的IMF组分
    #     axes[iter + 1][0].plot(imfs[iter])
    #     # 计算各组分的Hilbert变换
    #     imfsHT = hilbert(imfs[iter])
    #     # 绘制各组分的Hilber变换
    #     axes[iter + 1][0].plot(abs(imfsHT))
    #     # 设置图名
    #     axes[iter + 1][0].set_title('IMF{}'.format(iter))
    #     # 计算各组分Hilbert变换后的瞬时频率
    #     instf, timestamps = tftb.processing.inst_freq(imfsHT)
    #     # 绘制瞬时频率，这里乘以fs是正则化频率到真实频率的转换
    #     axes[iter + 1][1].plot(timestamps, instf * fs)
    #     # 计算瞬时频率的均值和中位数
    #     axes[iter + 1][1].set_title('Freq_Mean{:.2f}----Freq_Median{:.2f}'.format(np.mean(instf * fs), np.median(instf * fs)))
    # plt.show()


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
    attention_score = (features["beta_low"] + features["beta_high"]) / \
                      (features["alpha_low"] + features["alpha_high"] + features["theta"])
    return attention_score


def get_rhythm_features(oneframe, fs):
    '''
    提取信号节律波特征值
    :param oneframe: eeg信号
    :param fs: 采样频率
    :return: 特征值集合
    '''
    frameLength = len(oneframe)
    y_delta = sum(abs(rhythmExtraction(
        oneframe, 1, 3, fs, frameLength, 'δ wave'))) / frameLength
    y_theta = sum(abs(rhythmExtraction(
        oneframe, 4, 7, fs, frameLength, 'θ wave'))) / frameLength
    y_alpha1 = sum(abs(rhythmExtraction(
        oneframe, 8, 10, fs, frameLength, 'α_low wave'))) / frameLength
    y_alpha2 = sum(abs(rhythmExtraction(
        oneframe, 11, 13, fs, frameLength, 'α_high wave'))) / frameLength
    y_beta1 = sum(abs(rhythmExtraction(
        oneframe, 14, 20, fs, frameLength, 'β_low wave'))) / frameLength
    y_beta2 = sum(abs(rhythmExtraction(
        oneframe, 21, 30, fs, frameLength, 'β_high wave'))) / frameLength
    y_gamma1 = sum(abs(rhythmExtraction(
        oneframe, 31, 40, fs, frameLength, 'δ wave'))) / frameLength
    y_gamma2 = sum(abs(rhythmExtraction(
        oneframe, 41, 50, fs, frameLength, 'δ wave'))) / frameLength
    spectral_feature = {"delta": y_delta, "theta": y_theta, "alpha_low": y_alpha1,
                        "alpha_high": y_alpha2, "beta_low": y_beta1, "beta_high": y_beta2,
                        "gamma_low": y_gamma1, "gamma_high": y_gamma2}
    return spectral_feature


def get_meditation_score(features):
    '''
    获得当前帧的瞬时放松度
    :param oneframe: 一帧脑电数据
    :param fs: 信号采样率
    :return: 当前放松度得分
    '''
    meditation_score = (features["alpha_low"] + features["alpha_high"]) / \
                       (features["alpha_low"] + features["alpha_high"] +
                        features["theta"] + features["beta_low"] + features["beta_high"])
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


def wavplot(y_time, title):
    # 各类节律波可视化
    t = range(len(y_time))
    plt.plot(t, y_time)
    plt.title(title, fontsize=20)
    plt.show()
    return 0


def smooth_score(score_cache, observe_window, frame_window):
    '''
    attention得分平滑处理
    :param attention_cache: 历史attention得分
    :param observe_window: smooth时间窗口大小
    :param frame_window: 分帧的窗口大小
    :return: 输出有效attention得分
    '''
    if len(score_cache) > int(observe_window / frame_window):
        out_score = np.mean(score_cache[-int(observe_window / frame_window):])
    else:
        out_score = np.mean(score_cache)
    return out_score
