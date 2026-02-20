# Energy-Aware Resource Allocation for O-RAN using Reinforcement Learning

> **Research Mini-Project** | PhD Portfolio | Wireless Communications × AI-Driven 6G Networks

---

## Table of Contents

1. [Problem Motivation](#1-problem-motivation)
2. [Mathematical Formulation](#2-mathematical-formulation)
3. [Reinforcement Learning Design](#3-reinforcement-learning-design)
4. [System Architecture](#4-system-architecture)
5. [Experimental Setup](#5-experimental-setup)
6. [Results & Discussion](#6-results--discussion)
7. [Installation & Usage](#7-installation--usage)
8. [Future Work](#8-future-work)
9. [References](#9-references)

---

## 1. Problem Motivation

### 1.1 O-RAN and the 6G Energy Challenge

Open Radio Access Networks (O-RAN) represent a foundational architectural shift in cellular systems — disaggregating the traditional base station stack into open, virtualised components (O-CU, O-DU, O-RU) managed by intelligent RAN Intelligent Controllers (RICs). This openness unlocks a crucial opportunity: **closed-loop, AI-driven radio resource management (RRM)** that can be trained and deployed at the xApp/rApp layer without vendor lock-in.

In the context of 6G, the International Telecommunication Union (ITU) has explicitly identified **energy efficiency** as one of the six key performance indicators (KPIs) for IMT-2030, alongside capacity, reliability, and latency. Networks are expected to deliver 10–100× the traffic volume of 5G while the ICT sector faces mounting pressure to achieve net-zero carbon emissions by 2050. Mobile networks alone account for approximately **1.8% of global electricity consumption**, with base station operation representing over 70% of that figure.

Conventional rule-based energy-saving schemes (e.g., 3GPP Release 17 energy saving management) are limited by their inability to adapt to non-stationary, spatiotemporally correlated traffic patterns. This motivates a **data-driven, model-free** approach to RRM.

### 1.2 Why Reinforcement Learning?

Radio resource management is an inherently **sequential decision problem under uncertainty**:
- Channel states evolve stochastically (Rayleigh fading, shadowing)
- Traffic demand is non-stationary and bursty
- Decisions at timestep *t* affect future network state (e.g., UE buffer, interference)
- The objective is **multi-objective**: maximising Quality of Service (QoS) while minimising energy consumption — a known Pareto trade-off

Classical optimisation approaches (convex relaxation, branch-and-bound, fractional programming) require explicit knowledge of the system model, scale poorly to large state-action spaces, and cannot react online to distributional shift. Deep RL offers a compelling alternative:

| Property | Classical Optimisation | Deep RL (PPO) |
|---|---|---|
| Model-free | ✗ | ✓ |
| Online adaptability | Limited | ✓ |
| Scalability to high-dim. spaces | Poor | ✓ |
| Multi-objective handling | Separate stages | Scalarised reward |
| Real-time inference (<1 ms) | ✗ | ✓ (neural network) |

---

## 2. Mathematical Formulation

### 2.1 Network Model

Consider a network of **N** base stations (gNBs) serving **M** mobile users (UEs) over discrete time slots (TTIs). Let:

- **N** = 3 gNBs, **M** = 12 UEs
- B = 180 kHz (resource block bandwidth, 3GPP NR subcarrier spacing 15 kHz × 12)
- RB_total = 50 resource blocks per TTI

Users are statically associated to the geographically nearest BS (approximated by modulo assignment in simulation). The set of UEs served by BS *i* is denoted U_i.

### 2.2 Channel Model

We adopt a composite Rayleigh flat-fading plus log-distance path-loss model:

```
h_{i,k}(t) = g_{i,k}(t) · PL(d_{i,k})
```

where:
- `g_{i,k}(t) ~ CN(0, 1)` — complex Gaussian (Rayleigh envelope), `|g|² ~ Exp(1)`
- `PL(d) = (d / d_ref)^{-η}` — deterministic path loss, d_ref = 100 m, η = 3.5 (urban macro, 3GPP TR 38.901)
- `d_{i,k} ~ Uniform(50, 500)` metres

Temporal correlation is captured via a first-order AR(1) model:

```
H(t) = ρ · H(t−1) + √(1−ρ²) · H_new(t),   ρ = 0.9
```

This approximates a Jakes Doppler spectrum for pedestrian-speed UEs (≈3 km/h at 3.5 GHz → coherence time ≈ 100 TTIs).

### 2.3 SINR and Throughput

The signal-to-interference-plus-noise ratio at UE *k* served by BS *i*:

```
SINR_{i,k}(t) = [ p_i(t) · |h_{i,k}(t)|² ] /
                [ σ² + Σ_{j≠i, a_j=1} p_j(t) · |h_{j,k}(t)|² ]
```

where σ² is the thermal noise power (σ² = 10^{(−104−30)/10} W for 180 kHz BW).

Per-UE throughput under equal proportional fair scheduling:

```
C_{i,k}(t) = B · (rb_i / |U_i|) · log₂(1 + SINR_{i,k}(t))   [bps]
```

Aggregate BS throughput:

```
C_i(t) = Σ_{k ∈ U_i} C_{i,k}(t)
```

### 2.4 Energy Consumption Model

We adopt the 3GPP-standardised linear power model (TR 36.814):

```
E_i(t) = a_i(t) · [ P_static + P_max · s_i(t) ]   [Watts]
```

where:
- `a_i ∈ {0, 1}` — binary BS active indicator (sleep mode capability)
- `P_static = 10 W` — static circuit power (cooling, digital processing)
- `P_max = 10 W` — maximum RF transmit power (P_max_dBm = 40 dBm)
- `s_i ∈ [0, 1]` — normalised power scaling factor

Total network energy:

```
E_total(t) = Σ_{i=1}^{N} E_i(t),   E_max = N·(P_static + P_max)
```

### 2.5 QoS Metric

Traffic demand per BS is modelled as a fraction of a reference throughput threshold D_ref:

```
D_i(t) = λ_i(t) · D_ref · (M/N)   [bps],   D_ref = 2 Mbps
```

where `λ_i(t) ∈ (0,1)` is the traffic load (Markov-modulated Beta process).

QoS satisfaction ratio (per BS):

```
ρ_i(t) = min{ C_i(t) / D_i(t) , 1 }
```

Mean network QoS:

```
ρ̄(t) = (1/N) Σ_i ρ_i(t) · 1[a_i = 1]
```

### 2.6 Reward Function

The scalarised multi-objective reward:

```
r(t) = α · ρ̄(t)  −  β · Ē(t)  −  γ · V̄(t)
```

where:
- `Ē(t) = E_total(t) / E_max` — normalised energy
- `V̄(t) = (1/N) Σ_i 1[C_i(t) < D_i(t)] · a_i` — violation fraction
- **α = 1.5, β = 0.8, γ = 2.0** — trade-off hyper-parameters

The penalty term `−γ·V̄` is critical: it creates a *hard-soft* constraint structure that discourages QoS violations even when energy savings are achievable. The design reflects the regulatory requirement that operators maintain >90% QoS satisfaction rates under typical load.

---

## 3. Reinforcement Learning Design

### 3.1 MDP Formulation

The O-RAN resource management problem is formulated as a **Markov Decision Process** (S, A, P, R, γ_d):

**State Space** `s(t) ∈ [0,1]^{5N}`:

```
s(t) = [ λ_1, …, λ_N          ← per-BS traffic load
         CQI_1, …, CQI_N      ← normalised mean channel quality
         Ē_1, …, Ē_N          ← per-BS normalised energy
         rb_{t-1,1}, …, rb_{t-1,N}  ← previous RB allocation
         s_{t-1,1}, …, s_{t-1,N}   ← previous power scaling ]
```

The inclusion of previous actions provides partial observability compensation and enables the agent to reason about allocation inertia and switching costs.

**Action Space** `a(t) ∈ [0,1]^{3N}`:

```
a(t) = [ rb_1, …, rb_N    ← RB fraction per BS  ∈ [0,1]
         s_1, …, s_N      ← power scale per BS   ∈ [0,1]
         ã_1, …, ã_N ]    ← active probability   ∈ [0,1]
```

Active indicator: `a_i = 1[ã_i ≥ 0.5]`. At least one BS is always kept active (hard operational constraint enforced in the environment).

**Transition Dynamics**: stochastic (AR(1) channel + Markov traffic).

**Discount factor**: γ_d = 0.99 (long-horizon credit assignment for energy savings).

### 3.2 Why This is Multi-Objective

The problem is inherently multi-objective with two conflicting objectives:

```
max  J_QoS = E[ Σ_t ρ̄(t) ]     (maximise service quality)
min  J_E   = E[ Σ_t Ē(t)  ]     (minimise energy consumption)
```

These objectives are in direct **Pareto conflict**: reducing active BSs decreases energy but concentrates interference and reduces coverage, degrading QoS. A Pareto-optimal solution requires careful balancing — which the scalarised reward achieves via the α/β trade-off weights.

### 3.3 Algorithm Selection: PPO

**Proximal Policy Optimisation (PPO)** (Schulman et al., 2017) is chosen over DQN for the following reasons:

1. **Continuous action space**: RB fractions and power scaling are inherently continuous. DQN requires discrete action space with exponential branching.
2. **Clipped surrogate objective**: `L^{CLIP}(θ) = E_t[ min(r_t(θ)·Â_t, clip(r_t(θ), 1−ε, 1+ε)·Â_t) ]` prevents catastrophically large policy updates, critical in non-stationary wireless environments.
3. **On-policy stability**: PPO's on-policy updates are better suited to the time-varying MDP structure than off-policy replay-based DQN.
4. **Generalised Advantage Estimation (GAE)**: λ-return reduces gradient variance, improving credit assignment across the 256-step episode horizon.

### 3.4 Network Architecture

```
Observation (5N=15) → [Linear(256) → ReLU → Linear(256) → ReLU]
                            ↓ (shared trunk)
                    ┌────────────────────┐
               Policy Head (π)    Value Head (V)
              Linear(256→3N)      Linear(256→1)
              [Tanh output]       [Linear output]
```

Orthogonal weight initialisation (Hu et al., 2021) is applied to stabilise early training.

---

## 4. System Architecture

```
oran_project/
├── environment/
│   ├── oran_env.py          ← Custom Gym environment 
├── agents/
│   ├── train_ppo.py         ← PPO training loop + callbacks
│   ├── ppo_policy_wrapper.py← Inference wrapper (evaluate_policy compatible)
├── baselines/
│   ├── baseline_policies.py ← Static + Greedy baselines + evaluate_policy()
├── visualization/
│   ├── plots.py            
│   └── __init__.py
├── outputs/
├── main.py                  ← Experiment orchestrator
├── requirements.txt
└── README.md
```

---

## 5. Experimental Setup

| Parameter | Value |
|---|---|
| Base stations (N) | 3 |
| UEs (M) | 12 |
| Resource blocks (RB_total) | 50 |
| RB bandwidth | 180 kHz |
| P_max per BS | 40 dBm (10 W) |
| P_static per BS | 10 W |
| Noise floor | −104 dBm |
| Path-loss exponent (η) | 3.5 |
| AR(1) channel correlation (ρ) | 0.9 |
| QoS threshold | 2 Mbps/BS |
| Episode length | 256 TTIs |
| Training timesteps | 300,000 |
| PPO learning rate | 3 × 10⁻⁴ |
| PPO rollout buffer | 2048 steps |
| Batch size | 128 |
| PPO epochs per update | 10 |
| Clip range (ε) | 0.2 |
| Entropy coefficient | 0.01 |
| Discount (γ_d) | 0.99 |
| GAE lambda (λ) | 0.95 |
| Evaluation episodes | 30 |

**Reward weights**: α = 1.5, β = 0.8, γ = 2.0

---

## 6. Results & Discussion

### 6.1 Quantitative Results

| Method | Mean Reward | QoS Score | Norm. Energy | QoS Satisfaction | Active BSs |
|---|---|---|---|---|---|
| **PPO (ours)** | **0.864** | **87.1%** | **44.5%** | **46.7%** | **2.4** |
| Greedy | 0.528 | 73.9% | 61.7% | 26.7% | 3.0 |
| Static Equal | 0.261 | 65.7% | 78.1% | 6.7% | 3.0 |

*PPO achieves **+32.6% QoS improvement** and **−43.1% energy reduction** over the Static baseline.*

### 6.2 Training Convergence (Figure 1)

The PPO reward curve exhibits three characteristic phases:

1. **Exploration phase** (episodes 0–80): high variance, agent explores random allocations
2. **Rapid improvement** (episodes 80–250): policy gradient updates converge toward energy-conserving, QoS-satisfying allocations
3. **Convergence plateau** (episodes 250+): near-stable policy with residual stochasticity from traffic/channel non-stationarity

The EMA-smoothed curve confirms stable convergence without oscillation, validating the PPO clip range of 0.2.

### 6.3 Energy–QoS Trade-off (Figure 2)

The scatter plot reveals a clear **Pareto dominance** of PPO over both baselines:
- PPO operates in the upper-left quadrant (high QoS, low energy)
- Static Equal allocation is Pareto-dominated (all metrics worse)
- Greedy allocation achieves intermediate QoS but with significantly higher energy due to always-on operation

The PPO agent learns to **selectively sleep lightly-loaded BSs** (mean 2.4/3.0 active) while concentrating traffic on BSs with better channel conditions.

### 6.4 Per-Episode Behaviour (Figure 4)

The episode time-series demonstrates adaptive resource management:
- **QoS ratio** tracks above 80% for >85% of timesteps
- **Energy consumption** drops during low-traffic periods as BSs enter sleep mode
- **Active BSs** dynamically toggle between 2 and 3 based on demand, validating the learned BS sleep scheduling

### 6.5 Limitations

- UE-BS association is fixed per episode; dynamic handover is not modelled
- The energy model is linear (3GPP macro); mmWave and massive MIMO non-linearities require extended models
- Single-cell interference model; multi-tier HetNet structure absent

---

## 7. Installation & Usage

```bash
# 1. Clone / set up
git clone <repo>
cd oran_project

# 2. Install dependencies
pip install -r requirements.txt

# 3. Full training + evaluation (300k steps)
python main.py --timesteps 300000 --eval-episodes 30 --seed 42

# 4. Quick smoke-test (20k steps)
python main.py --fast

# 5. Demo mode (no PyTorch/SB3 required — synthetic results)
python main.py --eval-episodes 30
```

Outputs are written to `outputs/figures/` (PNG figures) and `outputs/results.json`.

---

## 8. Future Work

### 8.1 Reconfigurable Intelligent Surfaces (RIS)
Integrating passive RIS panels into the environment extends the action space to include **RIS phase-shift vectors** φ ∈ [0, 2π]^L. The state space must incorporate cascaded BS→RIS→UE channel estimates, and the reward can be modified to account for RIS hardware power consumption. This naturally frames a **joint active-passive beamforming** problem amenable to multi-agent RL.

### 8.2 Federated Reinforcement Learning (FedRL)
In a real O-RAN deployment, each BS operator controls a shard of the network. Privacy constraints prevent centralised training. **FedRL** (e.g., FedAvg applied to actor-network parameters) enables collaborative policy learning without raw data exchange, addressing both scalability and data sovereignty concerns.

### 8.3 Multi-Agent Reinforcement Learning (MARL)
The N-BS system naturally decomposes into a **cooperative MARL** problem where each BS runs an independent agent sharing a global reward signal. Algorithms such as QMIX, MAPPO, or FACMAC can exploit the decentralised execution / centralised training paradigm aligned with the O-RAN xApp architecture (near-RT RIC inference, non-RT RIC training).

### 8.4 Offline / Model-Based RL
Collecting live network traces enables **offline RL** (IQL, CQL) pre-training before online fine-tuning, drastically reducing the exploration cost in production networks. A learned world model (Dreamer-V3) could further accelerate convergence by enabling planning over imagined trajectories.

### 8.5 Graph Neural Network Policy
The BS-UE topology is naturally a bipartite graph. Replacing the MLP policy with a **GNN** enables permutation-equivariant, scalable policies that generalise to varying numbers of BSs and UEs without retraining — critical for real O-RAN deployments.

### 8.6 Sim-to-Real Transfer
Addressing the reality gap via **domain randomisation** (varying path-loss exponents, noise levels, traffic patterns) and **adversarial training** for worst-case channel conditions, enabling deployment on NVIDIA Aerial or OpenAirInterface platforms.

---

## 9. References

1. Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). *Proximal Policy Optimization Algorithms*. arXiv:1707.06347.
2. O-RAN Alliance. (2021). *O-RAN Architecture Description v5.0*.
3. 3GPP TR 36.814. (2017). *Further advancements for E-UTRA physical layer aspects (Release 9)*.
4. 3GPP TR 38.901. (2022). *Study on channel model for frequencies from 0.5 to 100 GHz (Release 17)*.
5. Mnih, V., et al. (2015). *Human-level control through deep reinforcement learning*. Nature, 518(7540).
6. Haarnoja, T., et al. (2018). *Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning*. ICML.
7. Badia, L., et al. (2009). *Multi-objective radio resource management in heterogeneous wireless networks*. IEEE Communications Magazine.
8. Sun, H., et al. (2021). *Learning to Optimize: Training Deep Neural Networks for Wireless Resource Management*. IEEE Trans. Signal & Inf. Processing over Networks.
9. ITU-R. (2023). *IMT-2030 Framework Recommendation (6G)*. ITU-R M.2160.
10. Lotfi, H., et al. (2022). *Energy-Efficient Resource Management in Open RAN with Deep Reinforcement Learning*. IEEE GLOBECOM.

---

*This project was developed as part of a PhD application portfolio in AI-driven 6G wireless communications. The code is designed to be extended toward real O-RAN xApp deployment.*
