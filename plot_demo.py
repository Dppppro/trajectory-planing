"""
可视化对比梯形曲线和 S 型曲线。

用法：
    python plot_demo.py
生成 result.png，并尝试弹出窗口显示。
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from trajectory import TrapezoidalProfile, SCurveProfile


def sample_all(profile, n=400):
    ts = np.linspace(0, profile.T, n)
    out = np.array([profile.sample(t) for t in ts])
    return ts, out[:, 0], out[:, 1], out[:, 2]


def main():
    q0, q1, vmax, amax, jmax = 0.0, 10.0, 2.0, 1.0, 5.0
    trap = TrapezoidalProfile(q0, q1, vmax, amax)
    scur = SCurveProfile(q0, q1, vmax, amax, jmax)

    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)

    for label, prof, color in (("Trapezoid", trap, "#185FA5"),
                               ("S-curve", scur, "#1D9E75")):
        t, q, v, a = sample_all(prof)
        axes[0].plot(t, q, color, label=label, lw=2)
        axes[1].plot(t, v, color, label=label, lw=2)
        axes[2].plot(t, a, color, label=label, lw=2)

    axes[0].set_ylabel("position")
    axes[0].set_title("Trajectory planning: trapezoid vs S-curve")
    axes[1].set_ylabel("velocity")
    axes[2].set_ylabel("acceleration")
    axes[2].set_xlabel("time (s)")
    axes[2].annotate("acceleration jumps -> mechanical shock",
                     xy=(trap.ta, -amax), xytext=(trap.T * 0.45, -amax * 0.75),
                     arrowprops=dict(arrowstyle="->", color="#A32D2D"),
                     color="#A32D2D", fontsize=10)
    for ax in axes:
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("result.png", dpi=120)
    print("saved: result.png")
    if matplotlib.get_backend().lower() != "agg":   # 无界面环境（如服务器）跳过弹窗
        plt.show()


if __name__ == "__main__":
    main()
