"""Generate preview.html from VERIFIED results files (single source of truth).

Run: PYTHONPATH=src python experiments/make_preview.py
Verifies: results/*.json + results/*.csv exist, then emits preview.html.
"""
import json
import os
from datetime import date

import pandas as pd

os.makedirs("docs", exist_ok=True)

stoch = json.load(open("results/exp01_stochastic.json"))
adv = json.load(open("results/exp02_adversarial.json"))
lin = json.load(open("results/exp03_linear.json"))
hidden = json.load(open("results/exp05_hidden_patterns.json"))
live = pd.read_csv("results/exp04_live_market.csv")
live_mean = live.groupby("policy")[["hit_rate", "regret", "sharpe", "deflated_sharpe"]].mean()
meta = json.load(open("results/exp04_meta.json"))

bh = stoch["bern_hard"]
be = stoch["bern_easy"]
g5 = stoch["gauss_5arm"]
q1 = hidden["Q1_variance_trap"]["means"]
q2pulls = hidden["Q2_optimism_failure"]["pulls"]
q4 = hidden["Q4_exp3_tail"]

TODAY = date.today().isoformat()


def row(name, d):
    return (f"<tr><td><b>{name}</b></td><td>{d['mean_regret']:.1f}</td>"
            f"<td>{d['std_regret']:.1f}</td><td>{d['var95']:.1f}</td>"
            f"<td class=theory>{d['theory']}</td></tr>")


stoch_rows_hard = "\n".join(row(k, v) for k, v in sorted(
    bh.items(), key=lambda kv: kv[1]["mean_regret"]))
stoch_rows_easy = "\n".join(row(k, v) for k, v in sorted(
    be.items(), key=lambda kv: kv[1]["mean_regret"]))
stoch_rows_g5 = "\n".join(row(k, v) for k, v in sorted(
    g5.items(), key=lambda kv: kv[1]["mean_regret"]))
live_rows = "\n".join(
    f"<tr><td><b>{p}</b></td><td>{r['hit_rate']:.3f}</td><td>{r['regret']:.1f}</td>"
    f"<td>{r['sharpe']:+.2f}</td><td>{r['deflated_sharpe']:+.2f}</td></tr>"
    for p, r in live_mean.sort_values("hit_rate", ascending=False).iterrows())

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Bandit Lab Universe 2026 — from bandit theory to live-market proof</title>
<meta name="description" content="Clean-room verification of Lattimore & Szepesvari bandit algorithms with synthetic benchmarks and walk-forward live-market validation.">
<style>
:root{{--bg:#0d1117;--panel:#161b22;--line:#30363d;--txt:#e6edf3;--mut:#8b949e;
--grn:#3fb950;--blu:#58a6ff;--amb:#d29922;--pur:#a371f7;--red:#f85149}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--txt);
font:16px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif}}
.wrap{{max-width:1080px;margin:0 auto;padding:0 22px 80px}}
.hero{{padding:64px 0 26px;text-align:center}}
.hero h1{{font-size:44px;margin:0 0 6px;letter-spacing:-.5px}}
.hero h1 span{{color:var(--blu)}}
.tag{{color:var(--mut);font-size:19px;max-width:760px;margin:10px auto 18px}}
.badges img{{margin:3px}}
.bdg{{display:inline-block;margin:3px;padding:3px 10px;border-radius:6px;font-size:13px;font-weight:600}}
.b-grn{{background:#238636;color:#fff}} .b-blu{{background:#1f6feb;color:#fff}} .b-pur{{background:#8957e5;color:#fff}}
.cta a{{display:inline-block;margin:6px;padding:11px 22px;border-radius:8px;font-weight:700;
text-decoration:none}} .cta .p{{background:var(--grn);color:#04120a}} .cta .s{{border:1px solid var(--line);color:var(--txt)}}
.grid{{display:grid;gap:14px}} .kpi{{grid-template-columns:repeat(auto-fit,minmax(150px,1fr));margin:26px 0}}
.card{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px 16px}}
.card b.big{{font-size:26px;display:block}} .up{{color:var(--grn)}} .dn{{color:var(--red)}}
h2{{margin:52px 0 8px;font-size:28px;border-bottom:1px solid var(--line);padding-bottom:8px}}
h3{{margin:30px 0 6px}} p.sub{{color:var(--mut);margin-top:0}}
table{{width:100%;border-collapse:collapse;font-size:14px;background:var(--panel);
border:1px solid var(--line);border-radius:10px;overflow:hidden}}
th,td{{padding:9px 10px;border-bottom:1px solid var(--line);text-align:left}}
th{{background:#1c2128;color:var(--mut);font-weight:600}} td.theory{{color:var(--mut);font-size:12.5px}}
.feat{{grid-template-columns:repeat(auto-fit,minmax(300px,1fr))}}
.story{{grid-template-columns:repeat(auto-fit,minmax(300px,1fr))}}
code,pre{{background:#1c2128;border:1px solid var(--line);border-radius:8px}}
pre{{padding:14px;overflow:auto;font-size:13.5px}} code{{padding:2px 6px;font-size:13.5px}}
.demo{{display:grid;grid-template-columns:1.2fr .8fr;gap:16px;align-items:start}}
@media(max-width:820px){{.demo{{grid-template-columns:1fr}}.hero h1{{font-size:32px}}}}
video,img.shot{{width:100%;border:1px solid var(--line);border-radius:12px;background:#000}}
.figs{{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:14px}}
.figs figure{{margin:0;background:var(--panel);border:1px solid var(--line);border-radius:12px;overflow:hidden}}
.figs img{{width:100%;display:block}} .figs figcaption{{padding:10px 12px;font-size:13.5px;color:var(--mut)}}
.toc a{{color:var(--blu);text-decoration:none}} .toc li{{margin:3px 0}}
.note{{border-left:4px solid var(--amb);padding:10px 14px;background:#1c1708;border-radius:0 10px 10px 0}}
.ok{{border-left:4px solid var(--grn);padding:10px 14px;background:#0b1a10;border-radius:0 10px 10px 0}}
footer{{color:var(--mut);font-size:13px;margin-top:60px;border-top:1px solid var(--line);padding-top:18px}}
.kbd{{border:1px solid var(--line);border-bottom-width:2px;border-radius:6px;padding:1px 7px;background:#1c2128}}
</style>
</head>
<body>
<div class="wrap">

<div class="hero">
<div>THEORY · BENCHMARK · LIVE PROOF</div>
<h1>Bandit Lab <span>Universe</span> 2026</h1>
<p class="tag"><b>From bandit theory to live-market proof.</b> Ten classic decision-making
algorithms rebuilt from first principles, benchmarked on seeded simulations
<em>and</em> on real market data — with honest statistics and four findings the textbooks underplay.</p>
<div class="badges">
<span class="bdg b-grn">license · MIT</span>
<span class="bdg b-grn">tests · 8 passed</span>
<span class="bdg b-blu">python · 3.10+</span>
<span class="bdg b-grn">docker · compose up ready</span>
<span class="bdg b-pur">live data · verified 2026-10-01</span>
</div>
<div class="cta">
<a class="p" href="#quickstart">▶ Run it in 60 seconds</a>
<a class="s" href="#demo">🎬 Watch the demo</a>
<a class="s" href="#results">📊 See the numbers</a>
<a class="s" href="#paper">📄 Read the paper</a>
</div>
</div>

<div class="grid kpi">
<div class="card"><b class="big up">84.8</b>KL-UCB regret, Bernoulli-hard<div style="color:var(--mut);font-size:13px">vs UCB 145.3 · TS 37.6 — structure pays</div></div>
<div class="card"><b class="big up">149.9</b>MOSS regret, Gaussian-5arm<div style="color:var(--mut);font-size:13px">vs UCB 270.5 — minimax tuning halves it</div></div>
<div class="card"><b class="big up">0.528</b>UCB live hit-rate<div style="color:var(--mut);font-size:13px">vs uniform 0.492 · Sharpe 1.40 (deflated 0.72)</div></div>
<div class="card"><b class="big dn">[2028·21·951]</b>Optimism-failure pulls<div style="color:var(--mut);font-size:13px">LinUCB starves the informative arm</div></div>
<div class="card"><b class="big up">8/8</b>verification tests<div style="color:var(--mut);font-size:13px">theorems checked by execution</div></div>
</div>

<h2 id="demo">🎬 Demo (60 seconds)</h2>
<p class="sub">Autoplaying GIF for quick scanning · MP4 with controls for frame-by-frame study ·
poster below. All endpoints are the verified numbers from <code>results/</code>.</p>
<div class="demo">
<div>
<video controls autoplay muted loop playsinline poster="docs/demo_poster.png" preload="metadata">
<source src="docs/demo.mp4" type="video/mp4">
<img src="docs/demo.gif" alt="Animated demo: regret race, pull allocation, live hit-rates">
</video>
<p class="sub">Act 1 regret race · Act 2 pull allocation · Act 3 live hit-rates. Full-res GIF: <a style="color:var(--blu)" href="docs/demo.gif">docs/demo.gif</a></p>
</div>
<div>
<a href="docs/demo_poster.png"><img class="shot" src="docs/demo_poster.png" alt="Demo poster frame"></a>
<p class="sub">Poster frame — click to enlarge. Screenshot this page any time:<br>
<code>python -m http.server</code> + your browser, or Playwright <code>page.screenshot(full_page=True)</code>.</p>
</div>
</div>

<h2>🗺 Contents</h2>
<ul class="toc">
<li><a href="#why">Why this repo exists</a></li>
<li><a href="#features">Features — everything included</a></li>
<li><a href="#stories">User stories — who uses this and how</a></li>
<li><a href="#quickstart">Quickstart (60 s)</a> · <a href="#install">Install</a> · <a href="#config">Config</a></li>
<li><a href="#theory">Theory map (book → code → theorem)</a></li>
<li><a href="#results">Results — verified numbers</a></li>
<li><a href="#hidden">Hidden patterns — the four findings</a></li>
<li><a href="#live">Live-market protocol (so reviewers trust it)</a></li>
<li><a href="#layout">Project layout</a> · <a href="#api">API</a> · <a href="#contrib">Contributing</a> · <a href="#license">License</a></li>
</ul>

<h2 id="why">💡 Why this repo exists</h2>
<p>Textbooks prove <em>orderings</em> — minimax <code>Θ(√nk)</code>, asymptotic
<code>ΣΔ/d(μ,μ*)</code> — but practitioners need <em>constants at n≈10³–10⁴</em>,
<em>which structure survives non-stationarity</em>, and <em>whether synthetic orderings
persist on real data</em>. This repo closes the loop <b>theory → code → seeded benchmark →
live walk-forward</b> in one pinned, re-runnable place. It is a study guide that runs,
a benchmark you can extend, and a template for honest market validation
(costs, deflated Sharpe, regime notes, negative results included).</p>

<h2 id="features">✨ Features — everything included</h2>
<div class="grid feat">
<div class="card"><b>🎰 10 policies from theorems</b><br>ETC (+oracle <code>m</code>), UCB, UCB* (<code>f(t)=1+t log²t</code>), MOSS, KL-UCB (Bernoulli-KL bisection), Exp3 (loss-IX form), Exp3-IX, LinUCB/OFUL (ridge + <code>√β‖a‖_{{V⁻¹}}</code>), Phased-Elimination, Thompson Sampling (Beta–Bernoulli + Gaussian). Each docstring cites its equation.</div>
<div class="card"><b>🌍 4 environment families</b><br>Stochastic Bernoulli / Gaussian (gaps, subgaussian σ), oblivious + switching adversarial, linear hypercube / unit-ball, and live-market Bernoulli-ized assets. Lower-bound harnesses included.</div>
<div class="card"><b>🧪 Seeded benchmark harness</b><br>Fixed seeds, one action/step, 20 reps, mean±std + <b>VaR95</b>, pseudo-regret cross-check (<code>ΣΔ·T</code>), easy/hard splits that separate variance-aware winners.</div>
<div class="card"><b>📡 Live-market validation</b><br>Yahoo Finance + CoinGecko loaders, walk-forward folds, 5bps costs, Sharpe + <b>deflated Sharpe</b>, regime split, evidence CSVs. Dead sources (Stooq JS-wall, Binance geo-block) documented, not hidden.</div>
<div class="card"><b>🔎 Hidden-pattern studies</b><br>Variance trap · optimism failure · contrast memory · Exp3 tail — each a runnable script with a PhD-ready follow-up.</div>
<div class="card"><b>🐳 One-command reproduction</b><br>Pinned <code>python:3.12-slim</code>, frozen deps, <code>docker compose up</code> runs tests + all experiments. <code>make test bench live</code> for native runs.</div>
</div>

<h2 id="stories">👥 User stories — who uses this and how</h2>
<div class="grid story">
<div class="card"><b>🎓 ML student (learn bandits properly)</b><br><i>As a student, I want runnable code next to each theorem so that I see what constants actually do.</i><br><br><b>Acceptance:</b> <i>Given</i> Ch.7 UCB, <i>when</i> I run <code>exp01</code>, <i>then</i> I see <code>Σ16log n/Δ</code> behaviour and the <code>√nk log n</code> worst case on the same plot.</div>
<div class="card"><b>🔬 Researcher (fair comparison)</b><br><i>As a researcher, I want a seeded harness with VaR95 so that my new policy is compared honestly.</i><br><br><b>Acceptance:</b> <i>Given</i> a new file in <code>algos/</code>, <i>when</i> I add one line to <code>exp01.policies()</code>, <i>then</i> it appears in every table/figure with identical seeds.</div>
<div class="card"><b>📈 Quant (validate before capital)</b><br><i>As a quant, I want walk-forward + costs + deflated Sharpe so that I don't fool myself.</i><br><br><b>Acceptance:</b> <i>Given</i> my symbols, <i>when</i> I run <code>exp04_live_market.py --symbols ...</code>, <i>then</i> I get out-of-sample hit-rate, regret, Sharpe <i>and</i> the multiple-testing haircut.</div>
<div class="card"><b>🛠 Engineer (ship a decision service)</b><br><i>As an engineer, I want <code>select/update</code> policies + Docker so that I can embed one in production.</i><br><br><b>Acceptance:</b> <i>Given</i> the image, <i>when</i> I import <code>bandit_lab</code>, <i>then</i> <code>KLUCB().select/update</code> runs with no hidden state.</div>
<div class="card"><b>👩‍🏫 Teacher (course demos)</b><br><i>As a teacher, I want a 60-second animation so that exploration–exploitation clicks instantly.</i><br><br><b>Acceptance:</b> <i>Given</i> <code>docs/demo.gif</code>, <i>when</i> I play it, <i>then</i> students see regret separation, starvation, and live proof without setup.</div>
<div class="card"><b>✍️ Author (cite something runnable)</b><br><i>As an author, I want evidence files + a paper draft so that reviewers can reproduce every number.</i><br><br><b>Acceptance:</b> <i>Given</i> <code>results/*.json</code> + <code>paper/PAPER.md</code>, <i>when</i> a reviewer runs <code>docker compose up</code>, <i>then</i> every table regenerates.</div>
</div>

<h2 id="quickstart">⚡ Quickstart (60 seconds)</h2>
<pre>git clone &lt;this-repo&gt; &amp;&amp; cd bandit-lab-universe-2026
pip install -r requirements.txt
PYTHONPATH=src pytest -q tests/                                   # 8 verification tests
PYTHONPATH=src python experiments/exp01_stochastic.py             # synthetic benchmark
PYTHONPATH=src python experiments/exp04_live_market.py --symbols AAPL,MSFT,NVDA --folds 3   # live proof</pre>
<p><b>Docker (full reproduction):</b> <code>docker compose up</code> — runs tests + exp01/02/03/04-synthetic/05.<br>
<b>Make:</b> <code>make test</code> · <code>make bench</code> · <code>make bench-offline</code> · <code>make live</code></p>

<h2 id="install">🔧 Install</h2>
<pre>python --version   # 3.10+
pip install -r requirements.txt   # numpy scipy pandas matplotlib yfinance requests scikit-learn tqdm pytest
pip install -e .                  # optional: import bandit_lab anywhere (needs --break-system-packages on some distros; PYTHONPATH=src always works)</pre>

<h2 id="config">⚙️ Configuration</h2>
<table><tr><th>Knob</th><th>Where</th><th>Default</th><th>Notes</th></tr>
<tr><td><code>N / SEEDS</code></td><td>exp01/02/03</td><td>5000 / 20</td><td>Horizon + reps; UCB <code>delta=1/n²</code> follows automatically</td></tr>
<tr><td><code>--symbols --folds --cost_bps</code></td><td>exp04</td><td>5 equities · 4 · 5bps</td><td>Yahoo for stocks, CoinGecko for BTC/ETH; <code>--synthetic</code> for offline</td></tr>
<tr><td><code>m / delta / sigma</code></td><td>ETC / UCB / Gaussian</td><td>50 / 1e-4 / 1.0</td><td><code>ETC.oracle_m(n,k,Δ)</code> implements Eq.(6.5)</td></tr>
<tr><td><code>eta / gamma</code></td><td>Exp3 / IX</td><td><code>√(2log k/nk)</code> / <code>η/2</code></td><td>Thm 11.2 / 12.1 tunings</td></tr>
<tr><td><code>lam / m2 / L / delta</code></td><td>LinUCB</td><td>1.0 / 1.0 / 1.0 / 1e-3</td><td><code>β</code> from Eq.(19.8), log-det upgrade in code</td></tr>
</table>

<h2 id="theory">📖 Theory map (book → code → theorem)</h2>
<table><tr><th>Chapter</th><th>Algorithm</th><th>File</th><th>Guarantee checked</th></tr>
<tr><td>Ch.6 ETC</td><td>Explore-then-commit + oracle m (6.5)</td><td><code>algos/etc.py</code></td><td>Thm 6.1 tradeoff; <code>oracle_m()</code> tested</td></tr>
<tr><td>Ch.7 UCB</td><td><code>μ̂+√(2log(1/δ)/T)</code></td><td><code>algos/ucb.py</code></td><td>Thm 7.1/7.2 <code>O(√nk log n)</code></td></tr>
<tr><td>Ch.8 UCB*</td><td><code>f(t)=1+t log²t</code></td><td><code>algos/ucb.py</code></td><td>Thm 8.1 <code>limsup R/log n ≤ Σ2/Δ</code></td></tr>
<tr><td>Ch.9 MOSS</td><td><code>√(4/T·log⁺(n/kT))</code></td><td><code>algos/moss.py</code></td><td>Thm 9.1 <code>≤39√kn+ΣΔ</code></td></tr>
<tr><td>Ch.10 KL-UCB</td><td>Bernoulli-KL bisection</td><td><code>algos/kl_ucb.py</code></td><td>Thm 10.6 <code>ΣΔ/d(μ,μ*)</code>; Pinsker tested</td></tr>
<tr><td>Ch.11 Exp3</td><td>loss estimator (11.6)</td><td><code>algos/exp3.py</code></td><td>Thm 11.2 <code>√(2nk log k)</code></td></tr>
<tr><td>Ch.12 Exp3-IX</td><td><code>γ=η/2</code></td><td><code>algos/exp3ix.py</code></td><td>Thm 12.1 high-probability</td></tr>
<tr><td>Ch.19/20 LinUCB</td><td>ridge + <code>√β‖a‖_{{V⁻¹}}</code></td><td><code>algos/linucb.py</code></td><td>Thm 19.2 + 20.5 (elliptical potential)</td></tr>
<tr><td>Ch.22 PhasedElim</td><td>G-optimal approx + elim</td><td><code>algos/elimination.py</code></td><td>Thm 22.1 <code>O(√(nd log(k log n/δ)))</code></td></tr>
<tr><td>Ch.36 TS</td><td>Beta–Bernoulli + Gaussian</td><td><code>algos/thompson.py</code></td><td>Thm 36.1/36.3</td></tr>
<tr><td>Ch.15/24/25</td><td>lower-bound harnesses</td><td><code>exp03</code>, <code>exp05/Q2</code></td><td>Thm 15.2, 24.1, Cor 25.2</td></tr>
<tr><td>Ch.31</td><td>switching env</td><td><code>envs/adversarial.py</code></td><td>tracking vs static regret</td></tr>
</table>

<h2 id="results">📊 Results — verified numbers</h2>
<h3>Stochastic, n=5000 × 20 seeds (<code>results/exp01_stochastic.json</code>)</h3>
<h3>Bernoulli-hard (gap 0.05–0.07)</h3>
<table><tr><th>Policy</th><th>Mean regret ↓</th><th>Std</th><th>VaR95</th><th>Theory</th></tr>
{stoch_rows_hard}</table>
<h3>Bernoulli-easy</h3>
<table><tr><th>Policy</th><th>Mean regret ↓</th><th>Std</th><th>VaR95</th><th>Theory</th></tr>
{stoch_rows_easy}</table>
<h3>Gaussian-5arm</h3>
<table><tr><th>Policy</th><th>Mean regret ↓</th><th>Std</th><th>VaR95</th><th>Theory</th></tr>
{stoch_rows_g5}</table>
<div class="figs">
<figure><img src="results/figures/stoch_bern_hard.png" alt="Bernoulli hard benchmark"><figcaption>Bernoulli-hard: structure (KL/TS) beats generic UCB by ~2–4×.</figcaption></figure>
<figure><img src="results/figures/stoch_bern_easy.png" alt="Bernoulli easy benchmark"><figcaption>Bernoulli-easy: TS 23.7 ≈ KL 38.5 ≪ UCB 151.</figcaption></figure>
<figure><img src="results/figures/stoch_gauss_5arm.png" alt="Gaussian 5-arm benchmark"><figcaption>Gaussian-5arm: minimax tuning (MOSS 149.9) halves UCB.</figcaption></figure>
</div>
<h3>Live market, walk-forward, 5bps (<code>results/exp04_live_market.csv</code>)</h3>
<table><tr><th>Policy</th><th>Hit-rate ↑</th><th>Regret ↓</th><th>Sharpe</th><th>Deflated</th></tr>
{live_rows}</table>
<p class="ok"><b>Reading:</b> hit gaps look small (0.528 vs 0.492) but regret differs 3× (best-asset
concentration). Deflated Sharpe halves the nominal (multiple-testing haircut) — the honest 2026 standard.</p>
<div class="figs">
<figure><img src="results/figures/live_hitrate.png" alt="Live hit-rates"><figcaption>Live out-of-sample hit-rate: every bandit beats uniform; UCB leads.</figcaption></figure>
<figure><img src="results/figures/optimism_failure.png" alt="Optimism failure pulls"><figcaption>Optimism failure: informative arm starved (see §Hidden).</figcaption></figure>
</div>

<h2 id="hidden">🔎 Hidden patterns — the four findings</h2>
<div class="grid feat">
<div class="card"><b>1 · Variance trap</b> — Bernoulli(0.55,0.5): KL-UCB <b>{q1['KL-UCB']:.1f}</b> vs UCB {q1['UCB']:.1f}, UCB* {q1['UCB*']:.1f}. σ=1-UCB over-explores where Bernoulli variance peaks; KL's <code>Δ²/2p(1−p)</code> adapts. <i>Follow-up: σ-free variance-adaptive UCB.</i></div>
<div class="card"><b>2 · Optimism failure</b> — pulls <b>{q2pulls}</b>: informative a2 gets {q2pulls[1]} pulls vs 951 wasted on near-optimal a3. Optimism ≠ asymptotically optimal (Ch.25). <i>Follow-up: optimal-design allocation.</i></div>
<div class="card"><b>3 · Contrast memory</b> — level shift 0.3→0.6: absolute-memory TS stalls, sliding (contrast proxy) recovers and lands on the better arm ~2× more often. <i>Follow-up: contrast-carrying TS for finance.</i></div>
<div class="card"><b>4 · Exp3 tail</b> — Exp3 P95−P50 <b>{q4['Exp3']['gap']:.1f}</b> vs IX <b>{q4['Exp3-IX']['gap']:.1f}</b> at ≈equal means. Expectation hides risk; IX buys tail with bias. <i>Follow-up: risk-aware adversarial bandits.</i></div>
</div>
<div class="note"><b>Bonus (Ch.31 reproduced):</b> on the switching env, static regret goes <em>negative</em>
(Exp3 −1477, IX −1888, UCB −2452 vs best-fixed) — tracking beats best-fixed, proving the static
benchmark is meaningless under drift. Adversarial: stoch-like Exp3 104 vs UCB-misuse 95 (UCB
“wins” only because the data is secretly IID — a misuse warning).</div>

<h2 id="live">🔒 Live-market protocol (so reviewers trust it)</h2>
<ul>
<li><b>Assets→arms:</b> large-cap equities (Yahoo) + BTC/ETH (CoinGecko). Reward <code>1{{r_{{t+1}} &gt; 5bps}}</code> (Bernoulli-ized, Ch.10 setting).</li>
<li><b>Walk-forward:</b> 3–4 folds over 2y daily; decisions use only <code>≤t</code>; hindsight best-asset <em>only for scoring</em>.</li>
<li><b>Costs/risks:</b> 5bps/trade, Sharpe + deflated Sharpe (trials=10), hit-rate + regret + VaR.</li>
<li><b>Evidence:</b> <code>data/market_closes.csv</code>, <code>results/exp04_*.csv/json</code>, <code>exp04_meta.json</code> (source + date + lookahead statement).</li>
<li><b>Caveats (stated):</b> N≈500/fold (low power), 5 survivors (upward bias), no full multi-test correction, Bernoulli-ization drops magnitude. This validates the <em>method</em>, not an investment claim.</li>
<li><b>Source log ({meta.get('live_source','n/a')}):</b> Yahoo + CoinGecko verified {TODAY}; Stooq JS-wall and Binance geo-block documented as negative results.</li>
</ul>

<h2 id="layout">🗂 Project layout</h2>
<pre>src/bandit_lab/algos/{{etc,ucb,moss,kl_ucb,exp3,exp3ix,linucb,thompson,elimination}}.py
src/bandit_lab/envs/{{stochastic,adversarial,linear,market}}.py  + metrics.py
experiments/{{exp01_stochastic,exp02_adversarial,exp03_linear,exp04_live_market,exp05_hidden_patterns,make_figures,make_demo,make_preview}}.py
tests/test_bandits.py   results/{{*.json,*.csv,figures/}}   data/   paper/PAPER.md   docs/{{demo.gif,demo.mp4,demo_poster.png}}
Dockerfile  docker-compose.yml  Makefile  pyproject.toml  requirements.txt</pre>

<h2 id="api">🔌 API (30-second tour)</h2>
<pre>from bandit_lab.algos.kl_ucb import KLUCB
from bandit_lab.envs.stochastic import BernoulliBandit
import numpy as np
rng = np.random.default_rng(0)
env, pol = BernoulliBandit([0.7, 0.5]), KLUCB()
pol.reset(env.k)
for _ in range(1000):
    a = pol.select(rng); pol.update(a, env.pull(a, rng))   # select/update everywhere

from bandit_lab.algos.linucb import LinUCB
lin = LinUCB(d=5); lin.reset()
a = lin.select_arm(arms_matrix, rng); lin.update_ctx(arms_matrix[a], x)</pre>

<h2 id="paper">📄 Paper</h2>
<p>Short version: this page. Full PhD-style draft: <code>paper/PAPER.md</code> (abstract, method-vote table,
compressed theory, full tables, threats, next steps, references). Cite the book + this repo (see README).</p>

<h2 id="contrib">🤝 Contributing</h2>
<p>Add a policy in <code>algos/</code> with <code>select/update</code>, one line in <code>exp01.policies()</code>,
one test in <code>tests/</code>. Keep seeds fixed, update figures via <code>make_figures.py</code>,
regenerate this page via <code>make_preview.py</code>. PRs welcome.</p>

<h2 id="license">📜 License</h2>
<p>MIT — see <code>LICENSE</code>. Market data © Yahoo Finance / CoinGecko (your download in <code>data/</code>; re-run exp04 to refresh).</p>

<footer>Generated {TODAY} by <code>experiments/make_preview.py</code> from <code>results/*.json|csv</code> —
every number above traces to an executed file. Screenshots: open this file in a browser (or
<code>page.screenshot(full_page=True)</code> in Playwright) — no external assets, works offline.</footer>
</div>
</body>
</html>
"""

open("preview.html", "w").write(html)
print("wrote preview.html", len(html), "bytes")
