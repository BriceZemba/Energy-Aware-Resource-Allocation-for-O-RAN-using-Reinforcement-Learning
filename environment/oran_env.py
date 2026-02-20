"""
oran_env.py  –  Energy-Aware O-RAN Resource Allocation Environment
===================================================================
A custom OpenAI Gym environment that abstracts a simplified Open RAN (O-RAN)
architecture for energy-aware radio resource management (RRM).

Mathematical Model
------------------
Network topology
    • N base stations (gNBs), M mobile users (UEs)
    • Users assigned to BSs using nearest-cell rule (randomised each episode)

Channel model (Rayleigh flat-fading + log-distance path loss)
    h_{i,k}(t) = g_{i,k}(t) · d_{i,k}^{-η/2}
    where g_{i,k}(t) ~ CN(0,1)  (i.i.d. complex Gaussian),
          d_{i,k}  ∈ [d_min, d_max]  (UE–BS distance in metres),
          η = 3.5                    (path-loss exponent, urban macro)

SINR at UE k, served by BS i:
    SINR_{i,k} = ( p_i · |h_{i,k}|² ) /
                 ( σ² + Σ_{j≠i, a_j=1} p_j · |h_{j,k}|² )

Shannon throughput (per UE, proportional RB share):
    C_{i,k} = B_RB · (rb_i / N_UE,i) · log₂(1 + SINR_{i,k})   [bps]
    Total BS throughput:  C_i = Σ_{k ∈ U_i} C_{i,k}

Energy model (3GPP-inspired linear model):
    E_i(t) = a_i · ( P_static + P_max · s_i )   [Watts]
    where a_i ∈ {0,1}  (BS active indicator),
          s_i ∈ [0,1]  (power scaling factor)

System-level QoS metric (satisfaction ratio ρ):
    ρ_i = min( C_i / D_i , 1 )     where D_i is demand (traffic-proportional)
    ρ̄   = (1/N) Σ_i ρ_i             (mean satisfaction across active BSs)

Reward function (multi-objective, scalarised):
    r(t) = α · ρ̄(t)  –  β · Ē(t)  –  γ · V(t)
    where Ē(t)  = Σ_i E_i(t) / E_max   (normalised total energy)
          V(t)  = (1/N) Σ_i 1[C_i < D_i]  (QoS violation fraction)
          α, β, γ  are trade-off hyper-parameters
"""

import numpy as np

# ── Minimal Gym-compatible stubs (no gym/gymnasium install required) ──────────
try:
    import gym
    from gym import spaces
    _GymBase = gym.Env
except ImportError:
    try:
        import gymnasium as gym
        from gymnasium import spaces
        _GymBase = gym.Env
    except ImportError:
        # Provide lightweight stubs so the environment is fully self-contained
        class _Box:
            def __init__(self, low, high, shape, dtype=float):
                self.low, self.high, self.shape, self.dtype = low, high, shape, dtype
        class _Spaces:
            Box = _Box
        spaces = _Spaces()
        class _GymBase:
            metadata = {}
            def reset(self):          raise NotImplementedError
            def step(self, action):   raise NotImplementedError
            def render(self, **kw):   pass
            def close(self):          pass


class ORANEnv(_GymBase):
    """
    Custom Gym environment modelling a simplified O-RAN system.

    Parameters
    ----------
    N_bs        : int   – Number of base stations
    N_ue        : int   – Number of mobile users (UEs)
    RB_total    : int   – Total resource blocks per TTI
    BW_per_RB   : float – Bandwidth per RB [Hz] (default 180 kHz, 3GPP NR)
    noise_dBm   : float – Thermal noise power [dBm]
    P_max_dBm   : float – Maximum transmit power per BS [dBm]
    P_static_W  : float – Static circuit power per BS [W]
    alpha       : float – QoS reward weight
    beta        : float – Energy penalty weight
    gamma       : float – QoS-violation penalty weight
    QoS_Mbps    : float – Per-BS minimum throughput threshold [Mbps]
    max_steps   : int   – Episode length (TTIs)
    seed        : int   – RNG seed
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        N_bs: int = 3,
        N_ue: int = 12,
        RB_total: int = 50,
        BW_per_RB: float = 180e3,
        noise_dBm: float = -104.0,
        P_max_dBm: float = 40.0,
        P_static_W: float = 10.0,
        alpha: float = 1.5,
        beta: float = 0.8,
        gamma: float = 2.0,
        QoS_Mbps: float = 2.0,
        max_steps: int = 256,
        seed: int = 42,
    ):
        super().__init__()

        # ── Network parameters ──────────────────────────────────────────────
        self.N = N_bs
        self.M = N_ue
        self.RB_total = RB_total
        self.BW = BW_per_RB                              # [Hz] per RB

        # ── Radio parameters ────────────────────────────────────────────────
        self.noise_power = 10 ** ((noise_dBm - 30) / 10)   # [W]
        self.P_max       = 10 ** ((P_max_dBm - 30) / 10)   # [W]
        self.P_static    = P_static_W                        # [W]
        self.eta         = 3.5                               # path-loss exponent

        # ── Reward hyper-parameters ──────────────────────────────────────────
        self.alpha = alpha
        self.beta  = beta
        self.gamma = gamma
        self.QoS_threshold = QoS_Mbps * 1e6               # [bps]

        # ── Episode control ──────────────────────────────────────────────────
        self.max_steps  = max_steps
        self.step_count = 0
        self.rng        = np.random.default_rng(seed)

        # ── Derived constants ────────────────────────────────────────────────
        self.E_max = self.N * (self.P_static + self.P_max)  # worst-case energy [W]

        # ── Action space ─────────────────────────────────────────────────────
        # Continuous: [ rb_fraction(N) | power_scale(N) | active_prob(N) ]
        # All ∈ [0, 1]; active thresholded at 0.5 inside step()
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(3 * self.N,), dtype=np.float32
        )

        # ── Observation space ─────────────────────────────────────────────────
        # [ traffic_load(N) | norm_CQI(N) | norm_energy(N) | prev_rb(N) | prev_pwr(N) ]
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(5 * self.N,), dtype=np.float32
        )

        # Initialise internal state
        self._init_state()

    # =========================================================================
    # Private helpers
    # =========================================================================

    def _init_state(self):
        """Initialise episode-level state variables."""
        self.traffic_load = self.rng.beta(2, 2, size=self.N)   # ∈ (0,1)
        self.channel      = self._sample_channel()
        self.prev_rb      = np.full(self.N, 0.5, dtype=np.float32)
        self.prev_power   = np.full(self.N, 0.5, dtype=np.float32)
        self.prev_active  = np.ones(self.N, dtype=np.float32)
        self.step_count   = 0

    def _sample_channel(self) -> np.ndarray:
        """
        Sample channel gain matrix H ∈ ℝ^{N×M}.
        h_{i,k} = |g_{i,k}|² · PL(d_{i,k})
        where g ~ CN(0,1) → |g|² ~ Exp(1)
        and   PL(d) = (d/d_ref)^{-η}  with d_ref = 100 m.
        """
        rayleigh   = self.rng.exponential(1.0, size=(self.N, self.M))
        distances  = self.rng.uniform(50, 500, size=(self.N, self.M))
        path_loss  = (distances / 100.0) ** (-self.eta)
        return rayleigh * path_loss                               # [dimensionless]

    def _evolve_channel(self):
        """
        Temporal channel evolution via first-order AR(1) model:
        H(t) = ρ·H(t-1) + √(1-ρ²)·H_new(t)    ρ = 0.9 (slow fading)
        """
        rho = 0.9
        H_new = self._sample_channel()
        self.channel = rho * self.channel + np.sqrt(1 - rho**2) * H_new

    def _evolve_traffic(self):
        """
        Traffic dynamics: Markov-modulated Poisson-like model.
        λ(t) = 0.7·λ(t-1) + 0.3·λ_new,   λ_new ~ Beta(2,2)
        """
        lambda_new        = self.rng.beta(2, 2, size=self.N)
        self.traffic_load = np.clip(0.7 * self.traffic_load + 0.3 * lambda_new, 0, 1)

    def _compute_throughput(
        self,
        rb_frac:    np.ndarray,
        pwr_scale:  np.ndarray,
        active_bin: np.ndarray,
    ) -> np.ndarray:
        """
        Compute per-BS aggregate Shannon throughput [bps].

        For each active BS i, UEs are partitioned equally among RBs
        (round-robin scheduling abstraction).  Co-channel interference from
        all simultaneously active BSs is included.
        """
        throughput = np.zeros(self.N, dtype=np.float64)

        for i in range(self.N):
            if active_bin[i] < 0.5:
                continue

            p_i   = pwr_scale[i] * self.P_max                  # Tx power [W]
            rb_i  = max(1, int(rb_frac[i] * self.RB_total))    # allocated RBs

            # UEs served by BS i (modulo assignment; stable across TTI)
            ue_set = [k for k in range(self.M) if k % self.N == i]
            if not ue_set:
                continue

            rb_per_ue = rb_i / len(ue_set)

            bs_tp = 0.0
            for k in ue_set:
                # Aggregate inter-cell interference
                I = sum(
                    pwr_scale[j] * self.P_max * self.channel[j, k]
                    for j in range(self.N)
                    if j != i and active_bin[j] >= 0.5
                )
                sinr     = (p_i * self.channel[i, k]) / (self.noise_power + I + 1e-15)
                sinr_db  = 10 * np.log10(max(sinr, 1e-9))

                # Capacity per UE per RB, accumulated across RBs
                bs_tp += self.BW * rb_per_ue * np.log2(1.0 + max(sinr, 1e-9))

            throughput[i] = bs_tp

        return throughput

    def _compute_energy(
        self,
        pwr_scale:  np.ndarray,
        active_bin: np.ndarray,
    ) -> np.ndarray:
        """
        Per-BS power consumption (W) using the 3GPP linear model:
            E_i = a_i · ( P_static + P_max · s_i )
        """
        return np.array(
            [active_bin[i] * (self.P_static + self.P_max * pwr_scale[i])
             for i in range(self.N)],
            dtype=np.float64,
        )

    def _compute_reward(
        self,
        throughput: np.ndarray,
        energy:     np.ndarray,
        active_bin: np.ndarray,
    ):
        """
        Scalarised multi-objective reward:
            r = α·ρ̄  –  β·Ē  –  γ·V̄

        Returns
        -------
        reward       : float
        qos_score    : float  – mean satisfaction ratio ρ̄ ∈ [0,1]
        energy_norm  : float  – normalised energy Ē ∈ [0,1]
        n_violations : int    – count of BSs below QoS threshold
        """
        # Demand per BS: proportional to traffic load × QoS threshold
        demand = self.traffic_load * self.QoS_threshold * (self.M / self.N)

        # Satisfaction ratio (capped at 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            sat = np.where(
                active_bin >= 0.5,
                np.minimum(throughput / (demand + 1e-12), 1.0),
                0.0,
            )
        qos_score = float(np.mean(sat))

        # Violation: active BSs not meeting demand
        violations    = int(np.sum((throughput < demand) & (active_bin >= 0.5)))
        violation_frac = violations / max(np.sum(active_bin >= 0.5), 1)

        # Normalised energy
        energy_norm = float(np.sum(energy) / (self.E_max + 1e-12))

        reward = (
            self.alpha * qos_score
            - self.beta  * energy_norm
            - self.gamma * violation_frac
        )

        return float(reward), qos_score, energy_norm, violations

    def _build_obs(
        self,
        rb_frac:   np.ndarray,
        pwr_scale: np.ndarray,
        energy:    np.ndarray,
    ) -> np.ndarray:
        """Assemble normalised observation vector."""
        cqi  = np.mean(self.channel, axis=1)
        cqi_norm  = cqi / (np.max(cqi) + 1e-9)
        e_norm    = energy / (self.E_max / self.N + 1e-9)
        e_norm    = np.clip(e_norm, 0, 1)

        obs = np.concatenate([
            self.traffic_load.astype(np.float32),
            cqi_norm.astype(np.float32),
            e_norm.astype(np.float32),
            rb_frac.astype(np.float32),
            pwr_scale.astype(np.float32),
        ])
        return np.clip(obs, 0, 1).astype(np.float32)

    # =========================================================================
    # Gym interface
    # =========================================================================

    def reset(self) -> np.ndarray:
        self._init_state()
        energy = self._compute_energy(self.prev_power, self.prev_active)
        return self._build_obs(self.prev_rb, self.prev_power, energy)

    def step(self, action: np.ndarray):
        action      = np.clip(action, 0.0, 1.0)
        rb_frac     = action[: self.N]
        pwr_scale   = action[self.N : 2 * self.N]
        active_prob = action[2 * self.N :]
        active_bin  = (active_prob >= 0.5).astype(np.float32)

        # Ensure at least one BS is active (operational constraint)
        if active_bin.sum() == 0:
            active_bin[np.argmax(active_prob)] = 1.0

        # Environment dynamics
        self._evolve_channel()
        self._evolve_traffic()

        # Physics
        throughput = self._compute_throughput(rb_frac, pwr_scale, active_bin)
        energy     = self._compute_energy(pwr_scale, active_bin)
        reward, qos_score, energy_norm, n_violations = self._compute_reward(
            throughput, energy, active_bin
        )

        # Next observation
        obs = self._build_obs(rb_frac, pwr_scale, energy)

        # Bookkeeping
        self.prev_rb     = rb_frac
        self.prev_power  = pwr_scale
        self.prev_active = active_bin
        self.step_count += 1
        done = self.step_count >= self.max_steps

        info = {
            "throughput_Mbps": throughput / 1e6,
            "energy_W":        energy,
            "total_energy_W":  float(np.sum(energy)),
            "qos_score":       qos_score,
            "energy_norm":     energy_norm,
            "n_violations":    n_violations,
            "active_bs":       int(active_bin.sum()),
            "traffic_load":    self.traffic_load.copy(),
        }

        return obs, reward, done, info

    def render(self, mode: str = "human"):
        active = int((self.prev_active >= 0.5).sum())
        print(
            f"[t={self.step_count:>4d}]  "
            f"Traffic: {self.traffic_load.round(2)}  "
            f"Active BSs: {active}/{self.N}  "
            f"Prev RB: {self.prev_rb.round(2)}  "
            f"Prev Pwr: {self.prev_power.round(2)}"
        )
