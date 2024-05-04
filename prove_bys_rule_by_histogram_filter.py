# 使用一维粒子直方图滤波算法证明贝叶斯概率必须融合先验概率p(x)，且必须使用运动模型更新先验
# 直方图滤波首先要离散化状态空间，
# 然后代入状态转移方程，更新先验概率
# 最后再使用量测模型更新概率
# 和粒子滤波的不一样在于，直方图滤波需要已知机器人的整个状态空间取值，如一维机器人位置可以离散化出来，
# 如下粒子的机器人在一维空间运动的定位精度为0.1m
# 但是粒子滤波适用于机器人状态空间未知，且无法穷举尽
import numpy as np
from matplotlib import pyplot as plt

door_pos = [1.0, 2.0, 5.0]
corridor_length = 4.8
# 0.1m划分一个区间
histgram_interval = 0.1 # localization accurancy
interval_num = corridor_length / histgram_interval
motion_step = 0.1
# 从0.5m处开始走
true_state = 0.5

interval_weight = [1.0]
interval_pos = [0.0]

# 假设机器人只能观测到距离其最近且位于其前面的门
def find_min_positive_value(obvs:list) -> float:
    min_positive = -1.0
    for obv in obvs:
        if obv > 0:
            min_positive = obv
    
    for i in range(len(obvs)):
        if obvs[i] > 0 and obvs[i] < min_positive:
            min_positive = obvs[i]
    
    # the robot can only see the door in front of it
    if min_positive < 0:
        return 1e20
    return min_positive

for i in range(1, int(interval_num)):
    interval_weight.append(1.0)
    interval_pos.append(interval_pos[i-1] + histgram_interval)

# change to numpy
interval_weight = np.array(interval_weight)

while true_state < corridor_length:
    # update current true state
    true_state += motion_step

    # prediction base on motion model
    # the weight should move with the robot
    # 预测模型的使用至关重要，必须依据当前的 p(x)来递推
    # interval_weight = np.roll(interval_weight, 1)
    interval_obv = []
    for i in range(len(interval_pos)):
        interval_pos[i] += motion_step
        obvs = np.array(door_pos) - interval_pos[i]
        ## 当前位置观测到其距离其最近门的距离
        min_positive_dist = find_min_positive_value(obvs)
        interval_obv.append(min_positive_dist)
    interval_obv = np.array(interval_obv) + 1e-5

    # 机器人只能观测到它前面距离其最近的一个door
    obvs = np.array(door_pos) - true_state
    min_positive_dist = find_min_positive_value(obvs)
    print("true min_positive_dist: ", min_positive_dist)

    # ok, now we have the measurement,
    # it's time to update the weight of histogram intervals
    diff = abs(interval_obv - min_positive_dist)
    obv_weights = 1.0 / diff

    # use bayes rule，p(x|y) ∝ p(x) * p(y|x)
    interval_weight *= obv_weights
    # no use bayes rule, p(x|y) = p(y|x), ERROR!!!
    # interval_weight = obv_weights
    # normalize weight
    interval_weight = interval_weight / np.sum(interval_weight)

    plt.plot(interval_pos, interval_weight)
    plt.scatter(np.array([true_state]), np.array([np.max(interval_weight)]), c='red' )
    plt.xlim((interval_pos[0], interval_pos[-1]))
    plt.ylim((0,1))
    plt.xticks(interval_pos[::3])
    plt.title("pos and weight")
    plt.show()
    plt.close()

