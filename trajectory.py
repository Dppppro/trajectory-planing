"""
轨迹规划：让电机在物理极限内平滑地走到目标位置。

为什么要规划：电机有最大速度和最大加速度，直接"全速冲过去再猛刹"
会过冲、会抖、会伤机械结构。轨迹规划就是先生成一条"每个时刻
位置/速度/加速度都确定且平滑"的路线，再交给 PID 去跟踪。

两种速度曲线：
1. TrapezoidalProfile  梯形曲线：加速 -> 匀速 -> 减速
                       实现最简单，但加速度有突变（理论上冲击无限大）
2. SCurveProfile       S 型曲线：七段式，加加速度(jerk)受限
                       起步和停止更柔，机械冲击小，高级伺服常用

两个类都提供 sample(t) -> (位置, 速度, 加速度)，单位任意（保持自洽即可）。
"""

import numpy as np


class TrapezoidalProfile:
    """梯形速度曲线规划器"""

    def __init__(self, q0, q1, vmax, amax):
        self.q0, self.q1 = float(q0), float(q1)
        self.dir = 1.0 if q1 >= q0 else -1.0
        D = abs(q1 - q0)
        self.vmax, self.amax = float(vmax), float(amax)

        d_acc = vmax * vmax / (2.0 * amax)      # 从 0 加速到 vmax 走过的距离
        if 2.0 * d_acc <= D:                     # 距离够长：梯形（有匀速段）
            self.ta = vmax / amax                # 加速段时长
            self.tc = (D - 2.0 * d_acc) / vmax   # 匀速段时长
            self.v_peak = vmax
        else:                                    # 距离太短：三角形（到不了 vmax）
            self.v_peak = np.sqrt(D * amax / 2.0)
            self.ta = self.v_peak / amax
            self.tc = 0.0
        self.T = 2.0 * self.ta + self.tc

    def sample(self, t):
        t = min(max(t, 0.0), self.T)
        if t <= 0.0:                          # 起步瞬间：位置在起点，速度/加速度为 0
            return self.q0, 0.0, 0.0
        if t >= self.T:                          # 已到达终点：静止
            return self.q1, 0.0, 0.0
        if t < self.ta:                          # 加速段
            a = self.amax
            v = self.amax * t
            q = 0.5 * self.amax * t * t
        elif t < self.ta + self.tc:              # 匀速段
            a = 0.0
            v = self.v_peak
            q = 0.5 * self.amax * self.ta ** 2 + self.v_peak * (t - self.ta)
        else:                                    # 减速段
            td = t - (self.ta + self.tc)
            a = -self.amax
            v = self.v_peak - self.amax * td
            q = (0.5 * self.amax * self.ta ** 2 + self.v_peak * self.tc
                 + self.v_peak * td - 0.5 * self.amax * td * td)
        return self.q0 + self.dir * q, self.dir * v, self.dir * a


class SCurveProfile:
    """S 型曲线规划器（七段式，jerk 受限）

    加速段由三个子段组成：jerk 正向（加速度 0->amax）-> 匀加速度 -> jerk 负向
    （加速度 amax->0）；中间可能有一段匀速；减速段是加速段的镜像。
    """

    def __init__(self, q0, q1, vmax, amax, jmax):
        self.q0, self.q1 = float(q0), float(q1)
        self.dir = 1.0 if q1 >= q0 else -1.0
        D = abs(q1 - q0)
        self.vmax, self.amax, self.jmax = float(vmax), float(amax), float(jmax)

        tj = self.amax / self.jmax               # 加加速度段时长
        v1 = 0.5 * self.jmax * tj * tj           # 单个 jerk 段获得的速度

        def accel_side(v_peak):
            """峰值速度为 v_peak 时，加速段的三相 [(时长, jerk)] 和走过的距离"""
            if v_peak > 2.0 * v1:                # 加速度能达到 amax
                t2 = (v_peak - 2.0 * v1) / self.amax   # 匀加速度段时长
                phases = [(tj, self.jmax), (t2, 0.0), (tj, -self.jmax)]
            else:                                # 加速度到不了 amax，jerk 段缩短
                ts = np.sqrt(v_peak / self.jmax)
                phases = [(ts, self.jmax), (0.0, 0.0), (ts, -self.jmax)]
            return phases, self._integrate(phases)

        d_full = accel_side(self.vmax)[1]
        if 2.0 * d_full <= D:                    # 距离够长：有匀速段
            self.v_peak = self.vmax
            self.tc = (D - 2.0 * d_full) / self.vmax
            accel_phases, _ = accel_side(self.v_peak)
        else:                                    # 距离太短：二分搜索合适的峰值速度
            lo, hi = 0.0, self.vmax
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                if 2.0 * accel_side(mid)[1] < D:
                    lo = mid
                else:
                    hi = mid
            self.v_peak = 0.5 * (lo + hi)
            self.tc = 0.0
            accel_phases, _ = accel_side(self.v_peak)

        # 减速段是加速段的时间镜像：jerk 序列倒序即可（无需取反，
        # 因为加速段本身就是 +jerk -> 0 -> -jerk，倒序后自然从 -jerk 开始减速）
        decel_phases = list(reversed(accel_phases))
        cruise = [(self.tc, 0.0)] if self.tc > 0 else []
        self.phases = accel_phases + cruise + decel_phases
        self.T = sum(d for d, _ in self.phases)

    @staticmethod
    def _integrate(phases):
        """按 [(时长, jerk)] 逐段积分，返回走过的距离"""
        a = v = q = 0.0
        for dur, jk in phases:
            q += v * dur + 0.5 * a * dur ** 2 + (1.0 / 6.0) * jk * dur ** 3
            v += a * dur + 0.5 * jk * dur ** 2
            a += jk * dur
        return q

    def sample(self, t):
        t = min(max(t, 0.0), self.T)
        a = v = q = 0.0
        for dur, jk in self.phases:
            if t >= dur:
                dt = dur
            else:
                dt = t
                q += v * dt + 0.5 * a * dt ** 2 + (1.0 / 6.0) * jk * dt ** 3
                v += a * dt + 0.5 * jk * dt ** 2
                a += jk * dt
                break
            q += v * dt + 0.5 * a * dt ** 2 + (1.0 / 6.0) * jk * dt ** 3
            v += a * dt + 0.5 * jk * dt ** 2
            a += jk * dt
            t -= dt
        return self.q0 + self.dir * q, self.dir * v, self.dir * a


def _check(prof, q0, q1, vmax, amax, jmax=None, n=2000):
    """对一个规划器做完整边界验证，失败就抛 AssertionError"""
    eps = 1e-6
    name = type(prof).__name__

    # 1) 起点：位置正确且静止
    q, v, a = prof.sample(0.0)
    assert abs(q - q0) < 1e-4, f"{name}: 起点位置 {q} != {q0}"
    assert abs(v) < 1e-4 and abs(a) < 1e-4, f"{name}: 起点未静止 v={v} a={a}"

    # 2) 终点：位置精确且静止
    q, v, a = prof.sample(prof.T)
    assert abs(q - q1) < 1e-4, f"{name}: 终点位置 {q} != {q1}"
    assert abs(v) < 1e-4 and abs(a) < 1e-4, f"{name}: 终点未静止 v={v} a={a}"

    # 3) 密集采样：约束不越界 + 位置单调 + 速度/加速度连续
    ts = [prof.T * i / (n - 1) for i in range(n)]
    qs = [prof.sample(t) for t in ts]
    assert max(abs(x[1]) for x in qs) <= vmax + eps, f"{name}: 速度越界"
    assert max(abs(x[2]) for x in qs) <= amax + eps, f"{name}: 加速度越界"
    direction = 1 if q1 >= q0 else -1
    for i in range(n - 1):
        dt = ts[i + 1] - ts[i]
        assert (qs[i + 1][0] - qs[i][0]) * direction >= -1e-9, f"{name}: 位置倒走"
        assert abs(qs[i + 1][1] - qs[i][1]) <= amax * dt + 1e-6, f"{name}: 速度跳变"
        if jmax is not None:   # S 曲线额外要求加速度连续（jerk 受限）
            assert abs(qs[i + 1][2] - qs[i][2]) <= jmax * dt + 1e-3, \
                f"{name}: 加速度跳变（jerk 越界）"

    print(f"  OK  {name:20s} 距离 {abs(q1 - q0):<6g} 总时长 {prof.T:6.2f}s")


def _self_check():
    """覆盖：常规 / 短距离 / 反向 / 零距离 / 极小 jerk 等边界"""
    print("梯形曲线:")
    _check(TrapezoidalProfile(0, 10, 2, 1), 0, 10, 2, 1)      # 常规：有匀速段
    _check(TrapezoidalProfile(0, 1, 2, 1), 0, 1, 2, 1)        # 短距离：退化成三角形
    _check(TrapezoidalProfile(10, 0, 2, 1), 10, 0, 2, 1)      # 反向
    _check(TrapezoidalProfile(5, 5, 2, 1), 5, 5, 2, 1)        # 零距离

    print("S 型曲线:")
    _check(SCurveProfile(0, 10, 2, 1, 5), 0, 10, 2, 1, 5)     # 常规：七段式
    _check(SCurveProfile(0, 0.5, 2, 1, 5), 0, 0.5, 2, 1, 5)   # 短距离：二分降速
    _check(SCurveProfile(0, 0.05, 2, 1, 5), 0, 0.05, 2, 1, 5) # 极短距离
    _check(SCurveProfile(10, 0, 2, 1, 5), 10, 0, 2, 1, 5)     # 反向
    _check(SCurveProfile(0, 10, 2, 1, 0.2), 0, 10, 2, 1, 0.2) # 极小 jerk
    _check(SCurveProfile(3, 3, 2, 1, 5), 3, 3, 2, 1, 5)       # 零距离

    print("\n全部通过：终点精确、起止静止、约束不越界、位置单调、jerk 受限")


if __name__ == "__main__":
    _self_check()