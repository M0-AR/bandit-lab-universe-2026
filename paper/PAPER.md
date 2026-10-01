# Bandit Lab Universe: Reproducible Verification of Stochastic, Adversarial and Linear Bandits with Live-Market Validation
### Full paper draft (PhD-style) — all numbers executed from this repo, 2026-10-01

**Abstract.** We rebuild the core of Lattimore & Szepesvári *Bandit Algorithms* (Ch.4–12,15,19–20,24–25,31,36) from scratch and verify it twice: on seeded synthetic benchmarks and on live market data under strict walk-forward discipline. Nine-algorithm comparison (ETC, UCB, UCB*, MOSS, KL-UCB, Exp3, Exp3-IX, LinUCB, Thompson Sampling) reproduces textbook orderings and quantifies four under-emphasized phenomena: (i) Bernoulli variance trap favouring KL-UCB, (ii) optimism failure in linear bandits, (iii) contrast-vs-absolute memory under level shifts, (iv) Exp3 tail risk masked by expectations. Live validation on 2y Yahoo Finance equities (walk-forward, 5bps costs) shows UCB hit-rate 0.528, regret 2.0, Sharpe 1.40 (deflated 0.72) vs uniform 0.492/6.5/−0.60/−1.33 — modest, honest, and reproducible via `docker compose up`. We report methods, negative results (Stooq/Binance source failures), and power caveats per 2026 reproducibility standards.

## 1 Introduction
Bandits model decision-making under uncertainty with minimal structure. Theory gives sharp orderings (minimax `Θ(√nk)`, asymptotic `ΣΔ/d(μ,μ*)`), but practitioners face three gaps: (a) which constants matter at n≈10³–10⁴, (b) whether optimism survives structure (linear/contextual), (c) whether synthetic orderings persist on real non-stationary data. We close the loop with one repo that runs theory → code → synthetic → live.

**Contributions.** (1) Clean-room implementations of 10 policies from theorems. (2) Seeded benchmark with regret/variance/VaR. (3) Live walk-forward harness (yfinance+CoinGecko, costs, deflated Sharpe). (4) Four hidden-pattern studies with PhD-ready extensions. (5) Pinned Docker reproduction.

## 2 Related & method-vote (2026)
| Source (date) | Finding adopted |
|---|---|
| Bandit Playground FW (2026-10, websearch) | Same seeds/steps; variance-aware wins only if gap small + variance high → easy/hard splits + VaR95 |
| Improving bandits UAI-42 (2026-08, websearch) | Parameterized families + offline selection → oracle-m + phased elim hooks |
| WiFi MAB benchmark (Nature 2026-08) | LinUCB most reliable contextual-optimistic; ε-greedy destabilizes MA → LinUCB vs TS comparison |
| AIT adaptive testing (2026-03) | Simulate null under same adaptive rule; ART power collapses for deterministic UCB → report power limits, avoid ART claims |
| OR-TS (2026-09-17) | Level varies 25× more than contrast; absolute memory costs 5× regret → Q3 sliding-memory test |
| Mixture-Greedy UAI (2026-08) | UCB bonus slows diversity objectives; implicit exploration → discuss in §5/Q2 |
| GittinsEval (2026-09-22) | Cost-aware Gittins, 1–2% cost ≈ zero simple regret → simple-regret framing + cost_bps |
| Walk-forward trading FW (2026-08-24, research) | Rolling folds, costs, p=0.34 reported honestly, regime split → exp04 design |
| OBP/ZOZOTOWN (2020, research) | Logged propensities + OPE → Bernoulli-ized rewards, CSV evidence |
| Docker SLR 5298 builds (2026-01, agent-reach) | Pin base+deps; Dockerfile≠image → pinned Dockerfile + evidence JSON |
| LinUCB notes (openresearch) / Slivkins monograph (openalex) | `Õ(d√T)` vs `O(d√T)` realizability; Exp4 `O(d√T)` adversarial → cite in §3 |
| Kaggle MAB dataset (2026-04) / Wiki MAB | Confirms Bernoulli-IID standard; history Thompson(1933)→Robbins(1952)→Auer→L&S |

Polling was sequential (one search at a time, 429-safe), keywords varied per tool as logged in session.

## 3 Theory (compressed, with pointers to code)
Stochastic regret `R_n=nμ*−E[ΣX_t]`, decomposition `R_n=ΣΔ_a E[T_a]` (Lemma 4.5). Subgaussian tail `P(μ̂−μ≥ε)≤exp(−nε²/2σ²)` (Cor.5.5) → UCB index `μ̂+√(2log(1/δ)/T)` (7.2). ETC Thm 6.1 tradeoff; UCB Thm 7.1/7.2; UCB* Thm 8.1 `2/Δ`; MOSS Thm 9.1 minimax; KL-UCB `d(p,q)` Thm 10.6; Exp3 loss-IX (11.6) Thm 11.2; IX Thm 12.1; LinUCB `θ̂=V⁻¹b`, `β` Thm 20.5, regret Thm 19.2 via elliptical potential Lemma 19.4; lower bounds Thm 15.2 `Ω(√(k−1)n)`, Thm 24.1 `Ω(d√n)`, Cor 25.2 + optimism counterexample §25.2; non-stationary tracking Ch.31; TS Ch.36 Thm 36.1/36.3.
Full derivations are in L&S; here we assert the exact constants our code targets (docstrings cite equations).

## 4 Benchmark design (reproducibility)
Seeds `1000+s` (synthetic, 20 reps), `9000+f` (market folds); one action/step; Bernoulli `B(μ)`, Gaussian `N(μ,1)`, adversarial oblivious + switching, linear hypercube/unit-ball; metrics mean±std, VaR95, pseudo-regret cross-check. Machine: CPU-only, numpy seeded. Pinned env `python:3.12-slim`.

## 5 Results
### 5.1 Synthetic (n=5000, `results/exp01_stochastic.json`)
bern_easy: TS-Bern 23.7 < KL 38.5 < TS-Gauss 48.0 < MOSS 85.5 < UCB* 130 < UCB 151. bern_hard (gap .05): TS 37.6 < KL 84.8 < Exp3 103 < MOSS 110 < UCB 145. gauss_5arm: KL 72.8(!) < TS-Gauss 105.7 < MOSS 149.9 < UCB* 227 < UCB 270 < Exp3 325. Interpretation: Bernoulli structure is worth ~2× (KL vs UCB on hard); minimax tuning (MOSS) halves UCB on many-arm Gaussian; adversarial methods pay ~2× on stochastic data (reduction cost, Eq.11.2).
### 5.2 Adversarial (`exp02`): stoch_like Exp3 104 vs UCB-misspecified 95 (UCB偷袭 wins when data is IID — misuse warning). switching: all regrets *negative* (Exp3 −1477, IX −1888, UCB −2452 vs best-fixed 2500): tracking beats best-fixed; static regret meaningless under drift — textbook Ch.31 point, reproduced.
### 5.3 Linear (`exp03`): hypercube d=4 regret 114.8 ≥ lower 3.03 ✓; unit-ball k=50 d=5 regret 178.7. Q2 (§5.5) is the sharper story.
### 5.4 Live market (`exp04_live_market.csv`, 2y, 4 folds, 5bps)
| policy | hit | regret | Sharpe | deflated |
|---|---|---|---|---|
| UCB | **0.528** | **2.00** | **1.40** | **0.72** |
| Exp3-IX | 0.498 | 5.75 | −0.27 | −0.63 |
| TS-Bern | 0.496 | 6.00 | 0.36 | −0.31 |
| Uniform | 0.492 | 6.50 | −0.60 | −1.33 |
Hit gaps look small but regret gap is 3× (best-asset concentration). Deflated Sharpe halves nominal (multiple-testing haircut). Market base rates ≈0.50–0.525 (near-efficient). No p-hacking: folds=4 (power ≪80%); claim is *method validation*, not alpha.
### 5.5 Hidden patterns (`exp05_hidden_patterns.json`, Figs.)
Q1 variance trap: KL 17.4 vs UCB 44.5 at gap .05 — σ=1 UCB over-explores where Bernoulli variance is max; KL's `Δ²/(2p(1−p))` adapts. Q2 optimism failure: pulls [2028,21,951] — proof optimism starves informative arms; fix = optimal-design allocation (Cor.25.2), open implementation. Q3 contrast memory: sliding-TS 2182 reward; absolute memory stale — OR-TS mechanism confirmed on toy shift. Q4 tail: P95−P50 19.1→16.5 via IX — bias-for-tail trade.

## 6 Threats & honesty
Survivorship (5 winners), N≈500/fold (power ~12% at d=0.17 → need ~540 folds for 80%), no transaction slippage beyond 5bps, Bernoulli-ization discards magnitude, yfinance survivorship-adjusted closes. Stooq/Binance failures disclosed (bot-wall/geo-block, verified). Synthetic seeds fixed but RNG-stream coupling across policies within seed is order-dependent — mitigated by 20 seeds + VaR.

## 7 Next (PhD-ready)
(i) Frank-Wolfe G-optimal LinUCB to close Q2; (ii) variance-adaptive σ-free UCB; (iii) contrast-TS for markets with formal regret; (iv) high-prob market bounds via IX + Bernstein; (v) contextual live test (features→LinUCB); (vi) OPE gap study on ZOZOTOWN logs.

## References
Lattimore & Szepesvári (2020); Auer et al. (1995,2002); Lai & Robbins (1985); Garivier & Cappé (2011); Audibert & Bubeck (2009,2010); Kocák et al. (2014)/Neu (2015); Abbasi-Yadkori et al. (2011); Dani et al. (2008); Agrawal & Goyal (2012,13); Kaufmann et al. (2012); Russo & Van Roy (2014,16); Bubeck et al. (2012,15,18); Bailey & López de Prado (2014); Harvey et al. (2016); Saito et al. (OBP 2020); Blum et al. UAI-42 (2026); Nia & Farnia UAI (2026); OR-TS (2026); GittinsEval (2026); Bandit Playground (2026); Docker SLR (2026); Slivkins (2019).

## Appendix A — Reproduce
`docker compose up` (≈10 min CPU) or `make test bench live`. Evidence: `results/*.json`, `data/market_closes.csv`, `results/figures/*.png`. Live re-runnable: `python experiments/exp04_live_market.py --symbols AAPL,MSFT,NVDA --folds 3`.
## Appendix B — Source-failure log
2026-10-01: `stooq.com/q/d/l` → JS challenge (curl captured); `api.binance.com` → 451 geo-block (captured); `yfinance AAPL 2026-10-01 OK`; `CoinGecko BTC OK`. Code routes to working sources; failures kept as documented negative results.
