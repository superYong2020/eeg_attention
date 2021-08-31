# 脑电专注度检测

## 1. 理论依据

目前，应用最广泛的评价专注度方法是Pope等于1995年提出的根据alpha,beta，theta频段能量计算认知专注度的公式(Pope et al., 1995)。
即beta频段能量与alpha和theta频段能量之和的比值，根据现今对脑电信号的理解，这是因为人们在注意力集中或者警觉的情况下脑电信号主要表现为beta频段的信号，
而人们在静息态或者睡眠时主要表现为alpha或者theta甚至更低频率频段的信号，因此这个比值可以表征人们注意力专注程度[1]。

![计算公式](./images/attention_score.png)

[1]李翀.基于机器人辅助神经康复的患者训练参与度与专注度研究.2017.清华大学,PhD dissertation.


## 2. 算法pipeline
![img.png](images/pipeline.png)


```mermaid
flowchat
st=>start: 开始
op=>operation: My Operation
cond=>condition: Yes or No?
e=>end
st->op->cond
cond(yes)->e
cond(no)->op
&```


## 3. 接口

#### HHTFilter(eegRaw, componentsRetain)
    滤波，去除眼电干扰
    :param eegRaw: 分帧后的数据      
    :param componentsRetain: 需要保留的信号成分     
    :return: 去除眼电后的干净数据

#### get_attention_score(oneframe, fs)
    获得当前帧的瞬时专注度
    :param oneframe: 一帧脑电数据
    :param fs: 信号采样率
    :return: 当前专注度得分

#### smooth_atten_score(attention_cache, observe_window, frame_window):
    attention得分平滑处理
    :param attention_cache: 历史attention得分
    :param observe_window: smooth时间窗口大小
    :param frame_window: 分帧的窗口大小
    :return: 输出有效attention得分


