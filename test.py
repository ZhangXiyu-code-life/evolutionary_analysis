import numpy as np
import matplotlib.pyplot as plt

# 假设参数（根据实际情况调整）
Cf = 240  # 政府的成本
Ri = 400  # 政府基础设施补贴
Rj = 400  # 政府其他补贴
Ct = 360  # 政府景区开发投资
Mt = 120  # 政府引导作用
Mg = 200  # 政府社会形象
Mup = 200  # 政府惩罚机制
β = 0.1  # 居民参与度参数
σ = 120  # 政企勾结导致政府收获的不正当收益
Rs = 160  # 企业社会责任
Rl = 400  # 企业长期收益
Rv = 200  # 企业直接收益
Rp = 160  # 企业投资回报
Cd = 120  # 企业成本
Ml = 200  # 企业维护成本
α = 0.1  # 居民参与度系数
Cp = 120  # 企业维护成本支付
Rr = 160  # 居民回报
Mc = 200  # 居民文化影响
Rk = 400  # 居民生活改善
Cr = 120  # 居民成本
Mv = 120  # 居民社会影响
Rh = 200  # 居民幸福感


# 定义支付函数
def government_payoff(strategy_gov, strategy_ent, strategy_res):
    if strategy_gov == 'C' and strategy_ent == 'C' and strategy_res == 'C':
        return -Cf + Ri - Ct + Rj
    elif strategy_gov == 'C' and strategy_ent == 'C' and strategy_res == 'D':
        return -Cf + Ri - Ct + Rj - Mup + σ
    elif strategy_gov == 'C' and strategy_ent == 'D' and strategy_res == 'C':
        return -Cf + Ri - Ct + Rj + Mt - Mg - Mup
    elif strategy_gov == 'C' and strategy_ent == 'D' and strategy_res == 'D':
        return -Cf - Ct + Rj + Mt - Mg - Mup
    elif strategy_gov == 'D' and strategy_ent == 'C' and strategy_res == 'C':
        return -Cf + Ri
    elif strategy_gov == 'D' and strategy_ent == 'C' and strategy_res == 'D':
        return -Cf + Ri - Rj + σ
    elif strategy_gov == 'D' and strategy_ent == 'D' and strategy_res == 'C':
        return -Cf - Mup + Ri - Rj
    elif strategy_gov == 'D' and strategy_ent == 'D' and strategy_res == 'D':
        return -Cf - Mup
    else:
        return 0


def enterprise_payoff(strategy_gov, strategy_ent, strategy_res):
    if strategy_gov == 'C' and strategy_ent == 'C' and strategy_res == 'C':
        return (1 - α) * Ri + Rv + Rp - Cd - Cp
    elif strategy_gov == 'C' and strategy_ent == 'C' and strategy_res == 'D':
        return Ct + Rl - Rv - Cd - Cp + σ
    elif strategy_gov == 'C' and strategy_ent == 'D' and strategy_res == 'C':
        return Rs - Ml - Cd - Mt - Rv
    elif strategy_gov == 'C' and strategy_ent == 'D' and strategy_res == 'D':
        return Rs - Rv - Cd - Ml - Mt
    elif strategy_gov == 'D' and strategy_ent == 'C' and strategy_res == 'C':
        return (1 - α) * Rl + Rv + Rp - Cd - Cp
    elif strategy_gov == 'D' and strategy_ent == 'C' and strategy_res == 'D':
        return Rl + Rv - Cd - Cp + σ
    elif strategy_gov == 'D' and strategy_ent == 'D' and strategy_res == 'C':
        return Rs - Rv - Cd
    elif strategy_gov == 'D' and strategy_ent == 'D' and strategy_res == 'D':
        return Rs - Rv - Cd
    else:
        return 0


def resident_payoff(strategy_gov, strategy_ent, strategy_res):
    if strategy_gov == 'C' and strategy_ent == 'C' and strategy_res == 'C':
        return α * Ri + Rv + Mc + Rk + Rh - Mup
    elif strategy_gov == 'C' and strategy_ent == 'C' and strategy_res == 'D':
        return (1 - β) * Rr - Mc - Cr - Mv
    elif strategy_gov == 'C' and strategy_ent == 'D' and strategy_res == 'C':
        return Rr + Mc + Rk + Rh + Mg + Ml
    elif strategy_gov == 'C' and strategy_ent == 'D' and strategy_res == 'D':
        return Rr - Mc - Rk - Cr + Mg + Ml
    elif strategy_gov == 'D' and strategy_ent == 'C' and strategy_res == 'C':
        return Rr + Mc + Rk + Rh + α * Rl - Mv
    elif strategy_gov == 'D' and strategy_ent == 'C' and strategy_res == 'D':
        return (1 - β) * Rr - Mc - Cr - Mv
    elif strategy_gov == 'D' and strategy_ent == 'D' and strategy_res == 'C':
        return Mc + Rk
    elif strategy_gov == 'D' and strategy_ent == 'D' and strategy_res == 'D':
        return -Mc - Cr
    else:
        return 0


# 初始化策略概率（每个玩家选择合作的概率）
x_g = np.array([0.5, 0.5])  # 政府策略概率 [C, D]
x_d = np.array([0.5, 0.5])  # 企业策略概率 [C, D]
x_r = np.array([0.5, 0.5])  # 居民策略概率 [C, D]


# 计算每个玩家选择某个策略的支付
def get_payoff_matrix(payoff_function):
    matrix = np.zeros((2, 2, 2))  # 2x2x2矩阵，表示每个玩家的所有策略组合
    strategies = ['C', 'D']
    for i, gov in enumerate(strategies):
        for j, ent in enumerate(strategies):
            for k, res in enumerate(strategies):
                matrix[i][j][k] = payoff_function(gov, ent, res)
    return matrix


# 获取每个玩家的支付矩阵
gov_matrix = get_payoff_matrix(government_payoff)
ent_matrix = get_payoff_matrix(enterprise_payoff)
res_matrix = get_payoff_matrix(resident_payoff)


# 复制动态方程的更新
def replicator_dynamics(x, payoff_matrix, n=0.01):
    avg_fitness = np.sum(x * payoff_matrix)  # 计算平均适应度
    fitness = np.dot(payoff_matrix, x)  # 计算每个策略的适应度
    dx = x * (fitness - avg_fitness)  # 计算策略频率变化的速率
    return x + n * dx  # 更新策略的概率


# 仿真
iterations = 500
gov_path, ent_path, res_path = [], [], []

for t in range(iterations):
    # 计算每个玩家的策略变化
    x_g = replicator_dynamics(x_g, gov_matrix)
    x_d = replicator_dynamics(x_d, ent_matrix)
    x_r = replicator_dynamics(x_r, res_matrix)

    # 记录每次迭代的策略概率
    gov_path.append(x_g)
    ent_path.append(x_d)
    res_path.append(x_r)

# 绘制三方博弈的策略演化
plt.figure(figsize=(10, 6))

# 绘制政府的策略演化
plt.plot([p[0] for p in gov_path], label='政府策略 C 的概率', color='r')

# 绘制企业的策略演化
plt.plot([p[0] for p in ent_path], label='企业策略 C 的概率', color='g')

# 绘制居民的策略演化
plt.plot([p[0] for p in res_path], label='居民策略 C 的概率', color='b')

plt.xlabel('代数（迭代次数）')
plt.ylabel('策略概率')
plt.title('三方博弈的演化动态')
plt.legend()
plt.show()
