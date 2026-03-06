# Design: Section 4 — Current Work on Modelling

> **Report format:** Interim workshop paper, 4-8 pages (aim for 4), NeurIPS checklist aware.
> **Philosophy:** Less is more. Appendices for technical detail.

---

## 4.1 MAPPO Architecture (~1 paragraph in main text)

Standard MAPPO following CityLight \cite{citylight}. Each of the 12 intersections acts as
an independent agent sharing a common policy network (2-layer ReLU MLP, 128 hidden units)
that maps local observations (queue lengths, vehicle counts, mean speeds, current phase,
time-of-day from sumo-rl) to a softmax distribution over legal signal phases. A centralized
critic receives the concatenated observations from all agents to estimate state value during
training only. Observations are normalized via Welford running statistics. Full
hyperparameters in Appendix B.

## 4.2 Reward Design (~2 short paragraphs)

**Paragraph 1 — motivation (time-loss failure):**
Our initial approach used a simple time-loss reward (`r = -log(1 + delay)`), directly
targeting the proposal's primary metric. However, this sparse, single-objective signal
proved insufficient for convergence: MAPPO trained under this reward achieved
8,434 +/- 589 s person-time-loss — **worse than even the Fixed-Time baseline**
(8,040 +/- 564 s). The policy failed to discover useful signal coordination from delay
feedback alone.

**Paragraph 2 — multi-objective reward:**
We replaced this with a multi-objective reward combining delay reduction, throughput
encouragement, and a throughput-fairness term across intersections. Under this reward,
MAPPO improved to 7,101 +/- 248 s — a 16% reduction over the time-loss reward and a clear
improvement over Fixed-Time, though still 25% behind Max-Pressure. The richer reward
landscape provides gradient signal even when delay changes are small, enabling the policy to
learn meaningful coordination. Full equation and weight values in Appendix C.

## 4.3 Experimental Setup (~1 short paragraph)

Toronto OSM corridor, 12 signalized intersections, 70 TMC-calibrated car flows active
0--3600 s. Episodes run for up to 300 s at 5 s decision intervals; demand exhausts at ~285 s.
Evaluation uses 10 random seeds per controller (MAPPO, Fixed-Time, Max-Pressure). Training
runs for 30 episodes with observation normalization on an RTX 4070 Laptop GPU.

## 4.4 Results and Analysis

### Results Table (main text)

Mean +/- std across 10 random seeds (objective reward mode):

| Controller   | Person Time Loss (s) | Avg Trip Time (s) | Arrived Vehicles | Vehicle Delay Fairness (Jain) |
|-------------|---------------------:|-------------------:|-----------------:|------------------------------:|
| Max-Pressure | 5,659 +/- 190       | 24.1 +/- 0.6      | 119.8 +/- 9.8   | 0.589 +/- 0.037               |
| MAPPO        | 7,101 +/- 248       | 25.2 +/- 0.8      | 116.9 +/- 9.8   | 0.476 +/- 0.048               |
| Fixed-Time   | 7,452 +/- 320       | 26.4 +/- 0.8      | 112.1 +/- 6.7   | 0.585 +/- 0.032               |

Time-loss reward mode (for comparison, motivation for Section 4.2):

| Controller   | Person Time Loss (s) | Avg Trip Time (s) | Arrived Vehicles |
|-------------|---------------------:|-------------------:|-----------------:|
| Max-Pressure | 5,659 +/- 190       | 24.1 +/- 0.6      | 119.8 +/- 9.8   |
| MAPPO        | 8,434 +/- 589       | 26.7 +/- 1.3      | 115.3 +/- 14.6  |
| Fixed-Time   | 8,040 +/- 564       | 28.1 +/- 1.1      | 121.8 +/- 8.8   |

### Composite Figure

4-panel figure (reports/figures/interim_objective_results.png):
- Panel A: MAPPO training curve (mean global reward vs episode)
- Panel B: Person time loss boxplots by controller
- Panel C: Eval metrics normalized to Max-Pressure baseline
- Panel D: Context card

### Analysis (~2 paragraphs)

**Paragraph 1 — honest baseline story:**
Max-Pressure dominates all metrics under nominal conditions. This is expected: it is
provably throughput-optimal under stationary single-commodity demand, which is exactly this
benchmark. The Toronto corridor with 70 uniform car flows is the regime where
Max-Pressure's local queue-balancing heuristic is near-optimal. No controller reaches the
proposal's Jain fairness target of 0.8, indicating that delay inequality across vehicles is
inherent to this network topology and demand pattern, not controller-specific.

**Paragraph 2 — methodological improvement path:**
The 25% gap motivates concrete next steps: (1) warm-start MAPPO from Max-Pressure imitation
so it begins at least as good as the heuristic, (2) curriculum training with progressively
harder scenarios (demand spikes, incidents, multi-modal flows) where Max-Pressure's
single-commodity assumption breaks, (3) extend training beyond 30 episodes — the training
curve (Panel A) shows the policy improving but not converged. The multi-objective reward
already demonstrates that richer reward signals substantially improve learning; the next
leverage point is the training environment, not the reward.

---

## Appendices (do not count toward page limit)

### Appendix A: Metric Definitions

| Metric | Definition |
|--------|-----------|
| Person-time-loss (s) | Aggregated excess travel time over free-flow conditions (SUMO `timeLoss`), weighted by estimated person occupancy: 1.3 per car, 30 per transit vehicle |
| Vehicle delay Jain fairness | Jain's index \cite{jain1984} computed over per-vehicle final time-loss values for all vehicles that completed their trip during the episode |
| Throughput (arrived vehicles) | Number of vehicles that completed their trip within the episode |
| Avg trip time (s) | Mean travel time (depart to arrival) across all arrived vehicles |

### Appendix B: MAPPO Hyperparameters

| Parameter | Value |
|-----------|-------|
| Hidden dimension | 128 |
| Hidden layers | 2 (actor and critic) |
| Activation | ReLU |
| Actor learning rate | 3 x 10^-4 |
| Critic learning rate | 1 x 10^-3 |
| Optimizer | Adam |
| Discount (gamma) | 0.99 |
| GAE lambda | 0.95 |
| PPO clip epsilon | 0.2 |
| Entropy coefficient | 0.01 |
| Value loss coefficient | 0.5 |
| Gradient clip (max norm) | 0.5 |
| PPO epochs per update | 5 |
| Minibatch size | 512 |
| Observation normalization | Welford running mean/variance |

### Appendix C: Reward Function

**Objective reward mode** (used for main results):

```
r_i = -w_d * log(1 + delay_i) + w_t * log(1 + throughput_i) + w_f * J(throughputs)
```

where:
- `delay_i` = total waiting time on edges approaching intersection i
- `throughput_i` = vehicle count on edges approaching intersection i
- `J(throughputs)` = Jain's index over per-agent throughputs (shared across all agents)
- `w_d = 1.0`, `w_t = 1.0`, `w_f = 0.25`

**Time-loss reward mode** (initial approach, abandoned):

```
r_i = -w_d * log(1 + delay_i)
```

The log transform normalizes the delay scale for PPO stability while preserving monotonicity.

---

## Citations Needed

| Key | Reference |
|-----|-----------|
| citylight | Zeng et al. 2024 — CityLight (already in proposal) |
| mappo | Yu et al. 2022 — The Surprising Effectiveness of PPO in Cooperative MARL |
| ppo | Schulman et al. 2017 — Proximal Policy Optimization |
| sumo_home | Eclipse SUMO (already in proposal) |
| sumo_rl | sumo-rl package (already in proposal as dependency) |
| jain1984 | Jain, Chiu, Hawe 1984 — A Quantitative Measure of Fairness |
| max_pressure_optimal | Varaiya 2013 — Max Pressure Control of a Network of Signalized Intersections |
| gae | Schulman et al. 2016 — High-Dimensional Continuous Control Using GAE |

---

## NeurIPS Checklist Notes (for Section 4)

- **Item 7 (Error bars):** Results table reports mean +/- std across 10 seeds. Variance source = SUMO random seed (vehicle insertion jitter). Method = sample standard deviation.
- **Item 8 (Compute):** RTX 4070 Laptop GPU. Training: ~10 min for 30 episodes. Eval: ~10 min for 30 episodes (3 controllers x 10 seeds). Total compute for reported experiments: <1 GPU-hour.
- **Item 6 (Experimental details):** Full hyperparameters in Appendix B, reward in Appendix C.
- **Item 4 (Reproducibility):** Code available in repo, pixi tasks for exact reproduction.
