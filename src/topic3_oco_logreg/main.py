import numpy as np
import time
import matplotlib.pyplot as plt
from dataclasses import dataclass
from pathlib import Path

try:
    import cvxpy as cp

    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False
    print("Warning: CVXPY not available, using projected GD only for baseline.")


@dataclass(frozen=True)
class OCOData:
    A: np.ndarray  # (T, d)
    y: np.ndarray  # (T,)
    R: float


# (T3.2) 数据生成
def generate_oco(seed: int = 1, d: int = 50, T: int = 5000, R: float = 5.0, noise_std: float = 0.5) -> OCOData:
    rng = np.random.default_rng(seed)
    w_star = rng.standard_normal(d)
    norm = np.linalg.norm(w_star)
    w_star = w_star / (norm + 1e-12) * (R / 2.0)

    A = rng.standard_normal((T, d))
    eps = rng.normal(0.0, noise_std, size=T)
    linear = A @ w_star + eps
    y = np.sign(linear)
    y[y == 0] = 1.0  # 处理极少数的零

    return OCOData(A=A, y=y.astype(float), R=R)


# 辅助函数：ℓ2 球投影
def proj_l2_ball(w: np.ndarray, R: float) -> np.ndarray:
    norm = np.linalg.norm(w)
    if norm > R:
        return w * (R / norm)
    return w

# 辅助函数：逻辑损失梯度
def logistic_loss_grad(a: np.ndarray, y: float, w: np.ndarray):
    z = y * np.dot(a, w)
    exp_minus_z = np.exp(-z)
    loss = np.log(1.0 + exp_minus_z)
    sigmoid_minus_z = 1.0 / (1.0 + np.exp(z))
    grad = -y * a * sigmoid_minus_z
    return loss, grad


# (T3.3) 基线1: CVXPY
def compute_cvxpy_baseline(A: np.ndarray, y: np.ndarray, R: float):
    if not CVXPY_AVAILABLE:
        raise ImportError("CVXPY not installed.")
    T, d = A.shape
    w = cp.Variable(d)
    losses = cp.sum(cp.logistic(cp.multiply(-y, A @ w)))
    prob = cp.Problem(cp.Minimize(losses), [cp.norm(w, 2) <= R])
    prob.solve(verbose=False)
    if prob.status not in ["optimal", "optimal_inaccurate"]:
        raise ValueError("CVXPY failed to solve")
    min_cum_loss = prob.value
    print('CVXPY:',min_cum_loss)
    return min_cum_loss


# (T3.3) 基线2: 高精度投影梯度下降
def compute_projected_gd_baseline(A: np.ndarray, y: np.ndarray, R: float, max_iters: int = 3000, eta: float = 0.05):
    T, d = A.shape
    w = np.zeros(d)
    for _ in range(max_iters):
        total_grad = np.zeros(d)
        for t in range(T):
            _, g = logistic_loss_grad(A[t], y[t], w)
            total_grad += g
        w = proj_l2_ball(w - eta * (total_grad / T), R)

    min_cum_loss = sum(logistic_loss_grad(A[t], y[t], w)[0] for t in range(T))
    print('高精度投影梯度下降:', min_cum_loss)
    return min_cum_loss


# (T3.4) 算法1: Online Gradient Descent (OGD)
def run_ogd(data: OCOData, eta: float, min_cum_loss: float):
    T, d = data.A.shape
    w = np.zeros(d)
    cum_loss = 0.0
    regrets = []
    avg_regrets = []
    times = []
    start_time = time.time()

    for t in range(T):
        loss, grad = logistic_loss_grad(data.A[t], data.y[t], w)
        cum_loss += loss
        w = proj_l2_ball(w - eta * grad, data.R)

        regret = cum_loss - min_cum_loss
        regrets.append(regret)
        avg_regrets.append(regret / (t + 1))
        times.append(time.time() - start_time)

    total_runtime = time.time() - start_time
    return regrets, avg_regrets, times, total_runtime


# (T3.4) 算法2: Follow-the-Regularized-Leader (FTRL)
def run_ftrl(data: OCOData, eta: float, lam: float, min_cum_loss: float):
    T, d = data.A.shape
    gsum = np.zeros(d)
    w = np.zeros(d)
    cum_loss = 0.0
    regrets = []
    avg_regrets = []
    times = []
    start_time = time.time()

    for t in range(T):
        loss, grad = logistic_loss_grad(data.A[t], data.y[t], w)
        cum_loss += loss
        gsum += grad
        w_unproj = - (eta / (lam + 1e-12)) * gsum
        w = proj_l2_ball(w_unproj, data.R)

        regret = cum_loss - min_cum_loss
        regrets.append(regret)
        avg_regrets.append(regret / (t + 1))
        times.append(time.time() - start_time)

    total_runtime = time.time() - start_time
    return regrets, avg_regrets, times, total_runtime


# (T3.4) 算法3: AdaGrad-style OCO
def run_adagrad(data: OCOData, base_eta: float, eps: float = 1e-8, min_cum_loss: float = 0.0):
    T, d = data.A.shape
    w = np.zeros(d)
    sum_sq_grad = np.full(d, eps)  # 避免除零
    cum_loss = 0.0
    regrets = []
    avg_regrets = []
    times = []
    start_time = time.time()

    for t in range(T):
        loss, grad = logistic_loss_grad(data.A[t], data.y[t], w)
        cum_loss += loss

        sum_sq_grad += grad ** 2
        adaptive_lr = base_eta / np.sqrt(sum_sq_grad)
        w -= adaptive_lr * grad
        w = proj_l2_ball(w, data.R)

        regret = cum_loss - min_cum_loss
        regrets.append(regret)
        avg_regrets.append(regret / (t + 1))
        times.append(time.time() - start_time)

    total_runtime = time.time() - start_time
    return regrets, avg_regrets, times, total_runtime


# 主函数 - 整合所有部分并生成结果图
if __name__ == "__main__":
    # 设置随机种子（可选，提高可重复性）
    np.random.seed(42)

    # 生成数据 (T3.2)
    data = generate_oco(seed=1)
    print(f"数据生成完成: T={data.A.shape[0]}, d={data.A.shape[1]}, R={data.R}")

    # 计算离线最优累计损失 (T3.3 - 两个基线)
    print("正在计算离线基线（这可能需要几秒钟）...")
    if CVXPY_AVAILABLE:
        min_cum_loss_cvxpy = compute_cvxpy_baseline(data.A, data.y, data.R)
        print(f"CVXPY 离线最优累计损失: {min_cum_loss_cvxpy:.4f}")
    min_cum_loss_gd = compute_projected_gd_baseline(data.A, data.y, data.R)
    print(f"Projected GD 离线最优累计损失: {min_cum_loss_gd:.4f}")
    min_cum_loss = min_cum_loss_cvxpy if CVXPY_AVAILABLE else min_cum_loss_gd  # 使用可用最好的

    # 运行 OGD (T3.4 + T3.5)
    print("运行 OGD (eta=0.1)...")
    reg_ogd, avg_ogd, _, runtime_ogd = run_ogd(data, eta=0.01, min_cum_loss=min_cum_loss)
    print(f"OGD 完成 - 总运行时间: {runtime_ogd:.2f}s")
    print(f"OGD 最终累计遗憾: {reg_ogd[-1]:.2f}, 平均遗憾: {avg_ogd[-1]:.4f}")

    # 运行 FTRL (T3.4 + T3.5)
    print("运行 FTRL (eta=0.1, lam=1.0)...")
    reg_ftrl, avg_ftrl, _, runtime_ftrl = run_ftrl(data, eta=0.01, lam=0.5, min_cum_loss=min_cum_loss)
    print(f"FTRL 完成 - 总运行时间: {runtime_ftrl:.2f}s")
    print(f"FTRL 最终累计遗憾: {reg_ftrl[-1]:.2f}, 平均遗憾: {avg_ftrl[-1]:.4f}")

    # 运行 AdaGrad (T3.4 + T3.5)
    print("运行 AdaGrad (base_eta=1.0)...")
    reg_adagrad, avg_adagrad, _, runtime_adagrad = run_adagrad(data, base_eta=0.5, min_cum_loss=min_cum_loss)
    print(f"AdaGrad 完成 - 总运行时间: {runtime_adagrad:.2f}s")
    print(f"AdaGrad 最终累计遗憾: {reg_adagrad[-1]:.2f}, 平均遗憾: {avg_adagrad[-1]:.4f}")

    # 创建输出目录
    Path("figures").mkdir(exist_ok=True)

    # 绘制并保存曲线 (T3.5 - 包含所有算法)
    t_values = np.arange(1, 5001)

    plt.figure(figsize=(10, 6))
    plt.plot(t_values, reg_ogd, label='OGD', alpha=0.8)
    plt.plot(t_values, reg_ftrl, label='FTRL', alpha=0.8)
    plt.plot(t_values, reg_adagrad, label='AdaGrad', alpha=0.8)
    plt.xlabel('Time step t')
    plt.ylabel('Cumulative Regret')
    plt.title('Topic 3: Cumulative Regret vs Time')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('figures/cumulative_regret.png', dpi=200)
    print("累计遗憾曲线已保存: figures/cumulative_regret.png")

    plt.figure(figsize=(10, 6))
    plt.plot(t_values, avg_ogd, label='OGD', alpha=0.8)
    plt.plot(t_values, avg_ftrl, label='FTRL', alpha=0.8)
    plt.plot(t_values, avg_adagrad, label='AdaGrad', alpha=0.8)
    plt.xlabel('Time step t')
    plt.ylabel('Average Regret (Reg_T / T)')
    plt.title('Topic 3: Average Regret vs Time')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('figures/average_regret.png', dpi=200)
    print("平均遗憾曲线已保存: figures/average_regret.png")

    plt.figure(figsize=(10, 6))
    plt.plot(t_values, avg_ogd, label='OGD', alpha=0.8)
    plt.plot(t_values, avg_ftrl, label='FTRL', alpha=0.8)
    plt.plot(t_values, avg_adagrad, label='AdaGrad', alpha=0.8)
    plt.ylim(-0.5, 0.5)
    plt.xlabel('Time step t')
    plt.ylabel('Average Regret (Reg_T / T)')
    plt.title('Topic 3: Average Regret vs Time')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('figures/average_regret_detail.png', dpi=200)
    print("平均遗憾曲线已保存: figures/average_regret.png")


