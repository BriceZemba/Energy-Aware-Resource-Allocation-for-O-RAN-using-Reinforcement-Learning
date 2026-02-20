"""
plots.py  –  Research-Quality Visualisation for O-RAN RL Experiments
=====================================================================
Generates publication-ready figures:

  1. Training reward curve (smoothed with exponential moving average)
  2. Energy vs QoS trade-off scatter / Pareto frontier
  3. Bar comparison: PPO vs baselines (reward, QoS, energy, violations)
  4. Per-BS resource allocation heatmaps from a single evaluation episode
  5. Combined summary dashboard (2×2 grid, suitable for paper figures)
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")                      # headless – works in any environment
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
from scipy.ndimage import uniform_filter1d

# ── Global style ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "DejaVu Serif",
    "font.size":         11,
    "axes.titlesize":    12,
    "axes.labelsize":    11,
    "legend.fontsize":   9,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "figure.dpi":        150,
    "axes.grid":         True,
    "grid.alpha":        0.35,
    "grid.linestyle":    "--",
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

COLORS = {
    "ppo":    "#2166AC",
    "greedy": "#D6604D",
    "static": "#4DAC26",
    "accent": "#762A83",
}


def _ema(values: list | np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Exponential moving average smoothing for training curves."""
    s = np.array(values, dtype=float)
    out = np.zeros_like(s)
    out[0] = s[0]
    for i in range(1, len(s)):
        out[i] = (1 - alpha) * out[i - 1] + alpha * s[i]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: Training Reward Curve
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_curve(
    ep_rewards: list[float],
    save_path:  str = "outputs/figures/fig1_training_curve.pdf",
):
    """
    Plots the per-episode cumulative reward during PPO training with
    a raw trace (transparent) and a smoothed EMA overlay.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    rewards = np.array(ep_rewards)
    smooth  = _ema(rewards, alpha=0.08)
    eps     = np.arange(1, len(rewards) + 1)

    fig, ax = plt.subplots(figsize=(6.5, 3.8))

    ax.fill_between(eps, rewards, smooth, alpha=0.15, color=COLORS["ppo"])
    ax.plot(eps, rewards, alpha=0.3, linewidth=0.8, color=COLORS["ppo"], label="Raw episode reward")
    ax.plot(eps, smooth,  linewidth=2.0, color=COLORS["ppo"], label="EMA-smoothed (α=0.08)")

    ax.set_xlabel("Training Episode")
    ax.set_ylabel("Cumulative Episode Reward")
    ax.set_title("Figure 1 — PPO Training Convergence on O-RAN Environment")
    ax.legend(loc="lower right")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Annotate convergence region
    conv_start = int(0.7 * len(rewards))
    ax.axvspan(conv_start, len(rewards), alpha=0.06, color="green",
               label="Convergence region")
    ax.annotate(
        f"Converged region\n(ep {conv_start}+)",
        xy=(conv_start + (len(rewards) - conv_start) * 0.3, smooth[-1]),
        fontsize=8, color="green",
    )

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: Energy–QoS Trade-off Scatter Plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_energy_qos_tradeoff(
    results:   dict,          # {"PPO": {...}, "Greedy": {...}, "Static": {...}}
    save_path: str = "outputs/figures/fig2_energy_qos_tradeoff.pdf",
):
    """
    Energy vs QoS scatter with per-agent episode-level points and means.
    Illustrates the Pareto trade-off achieved by RL vs. baselines.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    color_map = {"PPO": COLORS["ppo"], "Greedy": COLORS["greedy"], "Static": COLORS["static"]}
    marker_map = {"PPO": "o", "Greedy": "s", "Static": "^"}

    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    for name, res in results.items():
        c   = color_map.get(name, "gray")
        m   = marker_map.get(name, "o")
        qos = np.array(res["all_qos"]) * 100          # percentage
        enr = np.array(res["all_energy"]) * 100        # percentage of E_max

        ax.scatter(enr, qos, alpha=0.4, s=30, color=c, marker=m)
        ax.scatter(
            res["mean_energy_norm"] * 100,
            res["mean_qos"] * 100,
            s=180, color=c, marker=m, edgecolors="black", linewidths=1.2,
            label=f"{name}  (μ QoS={res['mean_qos']*100:.1f}%, μ E={res['mean_energy_norm']*100:.1f}%)",
            zorder=5,
        )

    # Ideal corner annotation
    ax.annotate("← Ideal (low E, high QoS)", xy=(0.02, 0.98), xycoords="axes fraction",
                fontsize=8, color="darkgreen", va="top")

    ax.set_xlabel("Normalised Energy Consumption  [% of $E_{\\mathrm{max}}$]")
    ax.set_ylabel("QoS Satisfaction Ratio  [%]")
    ax.set_title("Figure 2 — Energy–QoS Trade-off: PPO vs. Baselines")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.set_xlim(left=0)
    ax.set_ylim(0, 105)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: Bar Comparison (4-metric panel)
# ─────────────────────────────────────────────────────────────────────────────

def plot_comparison_bars(
    results:   dict,
    save_path: str = "outputs/figures/fig3_comparison_bars.pdf",
):
    """
    4-panel bar chart comparing PPO vs. baselines on:
    (a) Mean Episode Reward
    (b) Mean QoS Score [%]
    (c) Mean Normalised Energy [%]
    (d) QoS Satisfaction Rate [%] (fraction of episodes with no violation)
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    agents = list(results.keys())
    colors = [COLORS.get(a.lower()[:3], "steelblue") for a in agents]

    metrics = {
        "(a) Mean Episode Reward":           [results[a]["mean_reward"]        for a in agents],
        "(b) Mean QoS Score [%]":            [results[a]["mean_qos"] * 100     for a in agents],
        "(c) Mean Norm. Energy [%]":         [results[a]["mean_energy_norm"]*100 for a in agents],
        "(d) QoS Satisfaction Rate [%]":     [results[a]["qos_satisfaction"]*100 for a in agents],
    }

    fig, axes = plt.subplots(1, 4, figsize=(13, 4.5))
    fig.suptitle("Figure 3 — Performance Comparison: PPO vs. Baselines", fontsize=12)

    for ax, (metric_name, values) in zip(axes, metrics.items()):
        bars = ax.bar(agents, values, color=colors, edgecolor="white",
                      linewidth=0.8, width=0.55)
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(values) * 0.02,
                f"{val:.2f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold",
            )
        ax.set_title(metric_name, fontsize=9.5, pad=8)
        ax.set_ylabel("")
        ax.set_ylim(0, max(values) * 1.18 if max(values) > 0 else 1)
        ax.tick_params(axis="x", labelsize=9)

    # Highlight energy bar – lower is better annotation
    axes[2].annotate("lower ↓ better", xy=(0.5, 0.97), xycoords="axes fraction",
                     ha="center", va="top", fontsize=7.5, color="gray")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4: Episode Time-Series (single agent roll-out)
# ─────────────────────────────────────────────────────────────────────────────

def plot_episode_timeseries(
    ep_data:   dict,           # keys: "qos", "energy", "active_bs", "traffic"
    agent_name: str = "PPO",
    save_path:  str = "outputs/figures/fig4_episode_timeseries.pdf",
):
    """
    Detailed per-timestep trace for one evaluation episode.
    Plots QoS satisfaction, energy consumption, active BSs, and traffic load.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    T = len(ep_data["qos"])
    t = np.arange(T)

    fig = plt.figure(figsize=(9, 7))
    gs  = gridspec.GridSpec(3, 1, hspace=0.45, figure=fig)

    # ── (top) QoS satisfaction ratio ───────────────────────────────────────
    ax0 = fig.add_subplot(gs[0])
    ax0.plot(t, np.array(ep_data["qos"]) * 100, color=COLORS["ppo"], lw=1.5)
    ax0.axhline(y=80, color="red", ls="--", lw=0.9, alpha=0.7, label="80% threshold")
    ax0.set_ylabel("QoS Ratio [%]")
    ax0.set_title(f"Figure 4 — Evaluation Episode Trace  ({agent_name})")
    ax0.set_ylim(0, 105)
    ax0.legend(loc="lower right", fontsize=8)

    # ── (mid) Normalised energy + traffic ──────────────────────────────────
    ax1 = fig.add_subplot(gs[1])
    ax1.plot(t, np.array(ep_data["energy"]) * 100, color=COLORS["greedy"],
             lw=1.5, label="Norm. Energy [%]")
    ax1_twin = ax1.twinx()
    ax1_twin.plot(t, ep_data["traffic"], color="gray", lw=1.0, ls=":",
                  alpha=0.8, label="Avg Traffic Load")
    ax1.set_ylabel("Norm. Energy [%]", color=COLORS["greedy"])
    ax1_twin.set_ylabel("Traffic Load", color="gray")
    ax1.legend(loc="upper left", fontsize=8)
    ax1_twin.legend(loc="upper right", fontsize=8)
    ax1.set_ylim(0, 100)

    # ── (bottom) Active BSs ─────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[2])
    ax2.step(t, ep_data["active_bs"], color=COLORS["accent"], lw=1.5, where="post")
    ax2.set_ylabel("Active BSs")
    ax2.set_xlabel("Timestep (TTI)")
    ax2.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.set_ylim(0, ep_data["active_bs"][0] + 1)

    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Utility: collect episode trace data from a policy roll-out
# ─────────────────────────────────────────────────────────────────────────────

def collect_episode_trace(policy, env) -> dict:
    """Run one episode and collect step-level metrics."""
    obs     = env.reset()
    done    = False
    qos, energy, active_bs, traffic = [], [], [], []

    while not done:
        action = policy.select_action(obs)
        obs, _, done, info = env.step(action)
        qos.append(info["qos_score"])
        energy.append(info["energy_norm"])
        active_bs.append(info["active_bs"])
        traffic.append(float(np.mean(info["traffic_load"])))

    return {
        "qos":       qos,
        "energy":    energy,
        "active_bs": active_bs,
        "traffic":   traffic,
    }
