# =============================================================================
# LookOut-inspired Mixed-Initiative Dataset + Generator (v1.0)
#
# Developed by DBbun LLC — January 2026
#
# License:
#   Apache License 2.0 (Apache-2.0)
#   https://www.apache.org/licenses/LICENSE-2.0
#
# Notes:
# - This project is inspired by the ideas in Eric Horvitz's CHI'99 LookOut work on
#   mixed-initiative user interfaces and decision-theoretic action under uncertainty.
# - This code and the generated synthetic datasets are NOT affiliated with Microsoft
#   Research or Eric Horvitz, and do not include any original LookOut code/data.
#
# Stable outputs in output/:
#  - mixed_initiative_traces.csv
#  - mixed_initiative_traces.jsonl
#  - mixed_initiative_traces.config.json
#  - mixed_initiative_traces.quality_report.txt
#
# Generated figures (PNG) in output/ if matplotlib is available:
#  - fig1_manual_invocation_summary.png         (Fig 1 analog; data-driven)
#  - fig2_explicit_agent_flow.png               (Fig 2 analog; data-driven)
#  - fig3_auto_scoping_policy_map.png           (Fig 3 analog; data-driven; legend fixed)
#  - fig4_threshold_action_vs_noaction.png      (Fig 4 analog)
#  - fig5_context_shifted_thresholds.png        (Fig 5 analog)
#  - fig6_thresholds_action_dialog_noaction.png (Fig 6 analog)
#  - fig7_dwell_time_vs_message_length.png      (Fig 7 analog)
#
# Core idea:
# - p_true is calibrated-by-construction via latent evidence_score
# - p_model is a distorted p_true in logit space (temperature + noise)
# - modalities/actions + analytic p* thresholds are logged per episode
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Tuple, Callable
import math
import random
import csv
import json
import os
from datetime import datetime

# Optional plotting (publication-style figures). If matplotlib isn't installed, figures are skipped.
try:
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:
    HAVE_MPL = False


@dataclass
class EngineConfig:
    # Repro / size
    seed: int = 13
    n_users: int = 40
    n_episodes: int = 50_000

    # Output (stable file prefix; no version in filenames)
    output_dir: str = "output"
    file_prefix: str = "mixed_initiative_traces"

    # --- Fig 7: dwell time vs message length (sigmoid relationship) ---
    msg_len_min: int = 50
    msg_len_max: int = 2500
    dwell_min_sec: float = 2.0
    dwell_max_sec: float = 9.0
    dwell_center_bytes: float = 900.0
    dwell_width_bytes: float = 220.0
    dwell_noise_std: float = 0.6

    # When does the system "consider acting" relative to dwell/reading?
    early_check_prob: float = 0.35
    early_offset_std: float = 2.4
    late_offset_std: float = 1.8

    # --- Calibrated-by-construction posterior via latent evidence ---
    base_goal_rate: float = 0.18
    evidence_strength: float = 2.15
    evidence_noise_std: float = 0.35
    user_logit_shift_std: float = 0.20
    p_clip: float = 1e-4

    # --- Reported model probability p_model (distortion of p_true) ---
    model_logit_temperature: float = 1.05
    model_logit_noise_std: float = 0.20
    model_logit_bias: float = 0.0

    # --- Utility model (context dependent) ---
    u_no_action_goal: float = 0.35
    u_no_action_no_goal: float = 0.70

    u_action_goal: float = 0.95
    u_action_no_goal_base: float = 0.22

    u_dialog_goal: float = 0.75
    u_dialog_no_goal_base: float = 0.55

    # Fig 3: automated scoping
    u_scope_goal: float = 0.62
    u_scope_no_goal_base: float = 0.62

    # Context-dependent penalty when user is not ready (interrupt cost)
    penalty_action_no_goal_max: float = 0.55
    penalty_dialog_no_goal_max: float = 0.25
    penalty_scope_no_goal_max: float = 0.10

    # Additional penalty when urgency is low
    low_urgency_penalty_action: float = 0.12
    low_urgency_penalty_dialog: float = 0.05
    low_urgency_penalty_scope: float = 0.03

    # Additional reward when urgency is high
    high_urgency_bonus_action: float = 0.05
    high_urgency_bonus_dialog: float = 0.03
    high_urgency_bonus_scope: float = 0.02

    # --- Outcome simulation ---
    ignore_base: float = 0.05
    ignore_if_not_ready_boost: float = 0.35

    accept_if_goal_action: float = 0.90
    accept_if_no_goal_action: float = 0.10

    accept_if_goal_dialog: float = 0.82
    accept_if_no_goal_dialog: float = 0.18

    accept_if_goal_scope: float = 0.88
    accept_if_no_goal_scope: float = 0.35

    refine_if_accept: float = 0.35
    realized_utility_noise_std: float = 0.03

    # --- LookOut modality mixture (Fig 1/2/3) ---
    modality_manual_prob: float = 0.25
    modality_explicit_agent_prob: float = 0.45
    modality_auto_scoping_prob: float = 0.30

    # --- Baseline heuristics ---
    baseline_p_dialog: float = 0.30
    baseline_p_action: float = 0.70

    # --- Reporting / calibration ---
    calibration_bins: int = 10
    report_bins_target: int = 12
    report_min_bin_size: int = 500
    report_top_lines: int = 12
    not_ready_zero_epsilon: float = 1e-12

    slope_alpha: float = 0.05
    slope_warn_alpha: float = 0.20

    # Plots
    make_plots_if_possible: bool = True

    # Fig 3 policy map resolution
    fig3_bins_p: int = 25
    fig3_bins_nr: int = 20


# =============================
# Helpers
# =============================

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

def logit(p: float) -> float:
    p = min(max(p, 1e-12), 1.0 - 1e-12)
    return math.log(p / (1.0 - p))

def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))

def clip_p(p: float, eps: float) -> float:
    return min(max(p, eps), 1.0 - eps)

def safe_div(n: float, d: float) -> float:
    return n / d if d != 0 else float("nan")

def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")

def quantile(sorted_xs: List[float], q: float) -> float:
    if not sorted_xs:
        return float("nan")
    q = clamp01(q)
    pos = (len(sorted_xs) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_xs[lo]
    w = pos - lo
    return (1.0 - w) * sorted_xs[lo] + w * sorted_xs[hi]

def fmt_pct(x: float) -> str:
    if x != x:
        return "NaN"
    return f"{x*100:.2f}%"

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def outpath(cfg: EngineConfig, suffix: str) -> str:
    return os.path.join(cfg.output_dir, f"{cfg.file_prefix}.{suffix}")


# =============================
# Thresholds (Figs 4–6)
# =============================

def threshold_p_star(u1_goal: float, u1_nogoal: float, u2_goal: float, u2_nogoal: float) -> float:
    """
    Solve p* where EU1 == EU2:
      p*u1_goal + (1-p)*u1_nogoal = p*u2_goal + (1-p)*u2_nogoal
    """
    denom = (u1_goal - u1_nogoal) - (u2_goal - u2_nogoal)
    if abs(denom) < 1e-12:
        return 0.5
    p = (u2_nogoal - u1_nogoal) / denom
    return clamp01(p)


# =============================
# Stats
# =============================

def approx_auroc(scores: List[float], labels: List[int]) -> float:
    n = len(scores)
    if n == 0:
        return float("nan")
    n_pos = sum(labels)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    pairs = list(zip(scores, labels))
    pairs.sort(key=lambda x: x[0])

    ranks = [0.0] * n
    i = 0
    rank = 1
    while i < n:
        j = i
        while j + 1 < n and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg_rank = (rank + (rank + (j - i))) / 2.0
        for k in range(i, j + 1):
            ranks[k] = avg_rank
        rank += (j - i + 1)
        i = j + 1

    sum_ranks_pos = sum(ranks[idx] for idx, (_s, y) in enumerate(pairs) if y == 1)
    u = sum_ranks_pos - (n_pos * (n_pos + 1) / 2.0)
    return u / (n_pos * n_neg)

def calibration_table_and_ece(p: List[float], y: List[int], nbins: int) -> Tuple[List[Tuple[float, float, int]], float]:
    bins = [(i / nbins, (i + 1) / nbins) for i in range(nbins)]
    table: List[Tuple[float, float, int]] = []
    ece = 0.0
    n_total = len(p)
    if n_total == 0:
        return table, float("nan")
    for i, (lo, hi) in enumerate(bins):
        idxs = [j for j in range(n_total) if (p[j] >= lo and (p[j] < hi if i < nbins - 1 else p[j] <= hi))]
        n = len(idxs)
        if n == 0:
            continue
        avg_pred = sum(p[j] for j in idxs) / n
        emp = sum(y[j] for j in idxs) / n
        table.append((avg_pred, emp, n))
        ece += (n / n_total) * abs(emp - avg_pred)
    return table, ece

def normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

def two_sided_p_from_z(z: float) -> float:
    return 2.0 * (1.0 - normal_cdf(abs(z)))

def slope_test_binary_y(x: List[float], y: List[int]) -> Tuple[float, float, float]:
    """
    OLS slope test for y in {0,1}: y=a + b x + eps
    Returns (b, z, p_two_sided) using normal approximation.
    """
    n = len(x)
    if n != len(y) or n < 3:
        return float("nan"), float("nan"), float("nan")
    xbar = sum(x) / n
    ybar = sum(y) / n
    sxx = sum((xi - xbar) ** 2 for xi in x)
    if sxx <= 1e-18:
        return float("nan"), float("nan"), float("nan")
    sxy = sum((x[i] - xbar) * (y[i] - ybar) for i in range(n))
    b = sxy / sxx
    a = ybar - b * xbar
    rss = sum((y[i] - (a + b * x[i])) ** 2 for i in range(n))
    sigma2 = rss / (n - 2)
    se_b = math.sqrt(sigma2 / sxx) if sigma2 >= 0 else float("nan")
    if se_b <= 0 or se_b != se_b:
        return b, float("nan"), float("nan")
    z = b / se_b
    p = two_sided_p_from_z(z)
    return b, z, p


# =============================
# Robust equal-count binning for interpretability
# =============================

def equal_count_bins_from_sorted(sorted_rows: List[Dict[str, Any]], key: str, target_bins: int, min_bin_size: int) -> List[List[Dict[str, Any]]]:
    n = len(sorted_rows)
    if n == 0:
        return []
    ideal = max(min_bin_size, n // max(1, target_bins))
    bins: List[List[Dict[str, Any]]] = []
    i = 0
    while i < n:
        j = min(n, i + ideal)
        bins.append(sorted_rows[i:j])
        i = j
    if len(bins) >= 2 and len(bins[-1]) < min_bin_size:
        bins[-2].extend(bins[-1])
        bins.pop()
    return bins

def binned_rate_equal_count(rows: List[Dict[str, Any]], key: str, event_fn: Callable[[Dict[str, Any]], bool],
                            target_bins: int, min_bin_size: int) -> List[Tuple[float, float, int, float, float]]:
    if not rows:
        return []
    sr = sorted(rows, key=lambda r: float(r[key]))
    bins = equal_count_bins_from_sorted(sr, key, target_bins, min_bin_size)
    out: List[Tuple[float, float, int, float, float]] = []
    for b in bins:
        vals = [float(r[key]) for r in b]
        vmin, vmax = min(vals), max(vals)
        c = (vmin + vmax) / 2
        n = len(b)
        rate = safe_div(sum(1 for r in b if event_fn(r)), n)
        out.append((c, rate, n, vmin, vmax))
    return out

def binned_rate_not_ready_zero_collapse(rows: List[Dict[str, Any]], key: str, event_fn: Callable[[Dict[str, Any]], bool],
                                        target_bins: int, min_bin_size: int, zero_eps: float) -> List[Tuple[float, float, int, float, float]]:
    if not rows:
        return []
    zeros = [r for r in rows if float(r[key]) <= zero_eps]
    pos = [r for r in rows if float(r[key]) > zero_eps]
    out: List[Tuple[float, float, int, float, float]] = []
    if zeros:
        vals0 = [float(r[key]) for r in zeros]
        vmin0, vmax0 = min(vals0), max(vals0)
        c0 = (vmin0 + vmax0) / 2
        n0 = len(zeros)
        rate0 = safe_div(sum(1 for r in zeros if event_fn(r)), n0)
        out.append((c0, rate0, n0, vmin0, vmax0))
    if pos:
        sr = sorted(pos, key=lambda r: float(r[key]))
        bins = equal_count_bins_from_sorted(sr, key, target_bins, min_bin_size)
        for b in bins:
            vals = [float(r[key]) for r in b]
            vmin, vmax = min(vals), max(vals)
            c = (vmin + vmax) / 2
            n = len(b)
            rate = safe_div(sum(1 for r in b if event_fn(r)), n)
            out.append((c, rate, n, vmin, vmax))
    return out


# =============================
# Engine components
# =============================

class UserModel:
    """
    Latent evidence -> calibrated posterior p_true:
      evidence_score ~ N(0,1) (+ evidence_noise)
      p_true = sigmoid(base_logit + evidence_strength*e + user_shift)
      true_goal ~ Bernoulli(p_true)
    """
    def __init__(self, cfg: EngineConfig):
        self.cfg = cfg
        self.base_logit = logit(cfg.base_goal_rate)
        self.user_shift = {
            f"user_{i:03d}": random.gauss(0.0, cfg.user_logit_shift_std)
            for i in range(cfg.n_users)
        }

    def sample_user(self) -> str:
        return f"user_{random.randrange(self.cfg.n_users):03d}"

    def sample_evidence(self) -> float:
        cfg = self.cfg
        return random.gauss(0.0, 1.0) + random.gauss(0.0, cfg.evidence_noise_std)

    def p_true_from_evidence(self, user_id: str, evidence: float) -> float:
        cfg = self.cfg
        z = self.base_logit + cfg.evidence_strength * evidence + self.user_shift[user_id]
        return clip_p(sigmoid(z), cfg.p_clip)

    def sample_true_goal(self, p_true: float) -> int:
        return 1 if random.random() < p_true else 0


class AttentionModel:
    """
    Fig 7: dwell time increases sigmoidally with message length.
    """
    def __init__(self, cfg: EngineConfig):
        self.cfg = cfg

    def dwell_time(self, msg_len: int) -> float:
        cfg = self.cfg
        s = sigmoid((msg_len - cfg.dwell_center_bytes) / cfg.dwell_width_bytes)
        dwell = cfg.dwell_min_sec + (cfg.dwell_max_sec - cfg.dwell_min_sec) * s
        dwell += random.gauss(0.0, cfg.dwell_noise_std)
        return max(0.0, dwell)

    def time_since_focus(self, dwell: float) -> float:
        cfg = self.cfg
        if random.random() < cfg.early_check_prob:
            early = abs(random.gauss(0.0, cfg.early_offset_std))
            return max(0.0, dwell - early)
        late = max(0.0, random.gauss(0.0, cfg.late_offset_std))
        return dwell + late

    def not_ready(self, dwell: float, time_since_focus: float) -> float:
        cfg = self.cfg
        gap = max(0.0, dwell - time_since_focus)
        denom = (cfg.dwell_max_sec - cfg.dwell_min_sec) + 1e-9
        return clamp01(gap / denom)


class ObservationModel:
    """
    p_model = distorted(p_true) in logit space (temperature + bias + noise).
    """
    def __init__(self, cfg: EngineConfig):
        self.cfg = cfg

    def p_model(self, p_true: float) -> float:
        cfg = self.cfg
        z = logit(p_true)
        z = z / cfg.model_logit_temperature
        z = z + cfg.model_logit_bias
        z = z + random.gauss(0.0, cfg.model_logit_noise_std)
        return clip_p(sigmoid(z), cfg.p_clip)


def sample_time_urgency() -> float:
    u = random.random()
    return u ** 1.4


class UtilityModel:
    """
    Context-dependent utilities:
      - not_ready increases nuisance cost for interventions (no_goal utilities)
      - urgency increases reward for correct interventions (goal utilities) and reduces low-urgency nuisance
    """
    def __init__(self, cfg: EngineConfig):
        self.cfg = cfg

    def utilities(self, not_ready: float, urgency: float) -> Dict[str, Tuple[float, float]]:
        cfg = self.cfg
        u_no = (cfg.u_no_action_goal, cfg.u_no_action_no_goal)

        # attention-based nuisance penalties (applied to no_goal outcomes)
        u_action_nogoal = clamp01(cfg.u_action_no_goal_base - cfg.penalty_action_no_goal_max * not_ready)
        u_dialog_nogoal = clamp01(cfg.u_dialog_no_goal_base - cfg.penalty_dialog_no_goal_max * not_ready)
        u_scope_nogoal  = clamp01(cfg.u_scope_no_goal_base  - cfg.penalty_scope_no_goal_max  * not_ready)

        # low urgency adds additional nuisance for intervening
        low = (1.0 - urgency)
        u_action_nogoal = clamp01(u_action_nogoal - cfg.low_urgency_penalty_action * low)
        u_dialog_nogoal = clamp01(u_dialog_nogoal - cfg.low_urgency_penalty_dialog * low)
        u_scope_nogoal  = clamp01(u_scope_nogoal  - cfg.low_urgency_penalty_scope  * low)

        # high urgency boosts the benefit when correct
        u_action_goal = clamp01(cfg.u_action_goal + cfg.high_urgency_bonus_action * urgency)
        u_dialog_goal = clamp01(cfg.u_dialog_goal + cfg.high_urgency_bonus_dialog * urgency)
        u_scope_goal  = clamp01(cfg.u_scope_goal  + cfg.high_urgency_bonus_scope  * urgency)

        return {
            "no_action": (u_no[0], u_no[1]),
            "action": (u_action_goal, u_action_nogoal),
            "dialog": (u_dialog_goal, u_dialog_nogoal),
            "scope": (u_scope_goal, u_scope_nogoal),
        }


class LookOutPolicy:
    """
    Modality chooses allowed actions:
      - manual_invocation: {no_action, action} (Fig 1)
      - explicit_agent: {no_action, dialog, action} (Fig 2)
      - auto_scoping: {no_action, scope, dialog, action} (Fig 3)

    Chooses argmax EU, and stores analytic p* thresholds (Figs 4–6).
    """
    def __init__(self, cfg: EngineConfig):
        self.cfg = cfg

    def choose_modality(self) -> str:
        cfg = self.cfg
        r = random.random()
        if r < cfg.modality_manual_prob:
            return "manual_invocation"
        if r < cfg.modality_manual_prob + cfg.modality_explicit_agent_prob:
            return "explicit_agent"
        return "auto_scoping"

    def decide(self, p: float, utils: Dict[str, Tuple[float, float]], modality: str) -> Tuple[str, Dict[str, float], Dict[str, float]]:
        if modality == "manual_invocation":
            allowed = {"no_action", "action"}
            order = ["action", "no_action"]
        elif modality == "explicit_agent":
            allowed = {"no_action", "dialog", "action"}
            order = ["action", "dialog", "no_action"]
        else:
            allowed = {"no_action", "scope", "dialog", "action"}
            order = ["action", "scope", "dialog", "no_action"]

        eus: Dict[str, float] = {}
        for a in allowed:
            uG, uNG = utils[a]
            eus[a] = p * uG + (1 - p) * uNG

        thr: Dict[str, float] = {}
        thr["p_star_noaction_action"] = threshold_p_star(utils["no_action"][0], utils["no_action"][1],
                                                         utils["action"][0], utils["action"][1])
        thr["p_star_noaction_dialog"] = threshold_p_star(utils["no_action"][0], utils["no_action"][1],
                                                         utils["dialog"][0], utils["dialog"][1])
        thr["p_star_dialog_action"] = threshold_p_star(utils["dialog"][0], utils["dialog"][1],
                                                       utils["action"][0], utils["action"][1])
        thr["p_star_noaction_scope"] = threshold_p_star(utils["no_action"][0], utils["no_action"][1],
                                                        utils["scope"][0], utils["scope"][1])
        thr["p_star_scope_action"] = threshold_p_star(utils["scope"][0], utils["scope"][1],
                                                      utils["action"][0], utils["action"][1])
        thr["p_star_scope_dialog"] = threshold_p_star(utils["scope"][0], utils["scope"][1],
                                                      utils["dialog"][0], utils["dialog"][1])

        best = None
        best_eu = float("-inf")
        for a in order:
            if a in eus and eus[a] > best_eu + 1e-15:
                best = a
                best_eu = eus[a]
        if best is None:
            best = "no_action"

        return best, eus, thr


class OutcomeModel:
    """
    Samples user response {none, ignore, accept, reject} and realized utility.
    Ignore probability increases with not_ready.
    """
    def __init__(self, cfg: EngineConfig):
        self.cfg = cfg

    def ignore_prob(self, not_ready: float) -> float:
        cfg = self.cfg
        return clamp01(cfg.ignore_base + cfg.ignore_if_not_ready_boost * not_ready)

    def expected_utility_for_action(self, action: str, true_goal: int, not_ready: float,
                                    utils: Dict[str, Tuple[float, float]]) -> float:
        cfg = self.cfg
        u_no = utils["no_action"][0] if true_goal else utils["no_action"][1]
        if action == "no_action":
            return u_no

        ign = self.ignore_prob(not_ready)
        u_ignore = clamp01(u_no - (0.04 + 0.10 * not_ready))

        if action == "action":
            acc = cfg.accept_if_goal_action if true_goal else cfg.accept_if_no_goal_action
            u_accept = utils["action"][0]
            u_reject = utils["action"][1]
        elif action == "dialog":
            acc = cfg.accept_if_goal_dialog if true_goal else cfg.accept_if_no_goal_dialog
            u_accept = utils["dialog"][0]
            u_reject = utils["dialog"][1]
        elif action == "scope":
            acc = cfg.accept_if_goal_scope if true_goal else cfg.accept_if_no_goal_scope
            u_accept = utils["scope"][0]
            u_reject = utils["scope"][1]
        else:
            return u_no

        eu_not_ignored = acc * u_accept + (1 - acc) * u_reject
        eu = ign * u_ignore + (1 - ign) * eu_not_ignored
        return clamp01(eu)

    def sample(self, action: str, true_goal: int, not_ready: float,
               utils: Dict[str, Tuple[float, float]]) -> Tuple[str, float, int]:
        cfg = self.cfg
        u_no = utils["no_action"][0] if true_goal else utils["no_action"][1]

        if action == "no_action":
            return "none", u_no, 0

        ign = self.ignore_prob(not_ready)
        if random.random() < ign:
            realized = clamp01(u_no - (0.04 + 0.10 * not_ready))
            return "ignore", realized, 0

        if action == "action":
            acc = cfg.accept_if_goal_action if true_goal else cfg.accept_if_no_goal_action
            if random.random() < acc:
                refine = 1 if random.random() < cfg.refine_if_accept else 0
                return "accept", utils["action"][0], refine
            return "reject", utils["action"][1], 0

        if action == "dialog":
            acc = cfg.accept_if_goal_dialog if true_goal else cfg.accept_if_no_goal_dialog
            if random.random() < acc:
                refine = 1 if random.random() < cfg.refine_if_accept else 0
                return "accept", utils["dialog"][0], refine
            return "reject", utils["dialog"][1], 0

        if action == "scope":
            acc = cfg.accept_if_goal_scope if true_goal else cfg.accept_if_no_goal_scope
            if random.random() < acc:
                refine = 1 if random.random() < cfg.refine_if_accept else 0
                return "accept", utils["scope"][0], refine
            return "reject", utils["scope"][1], 0

        return "none", u_no, 0


# =============================
# Baselines (heuristics)
# =============================

def baseline_b0_no_action(p_model: float, cfg: EngineConfig) -> str:
    return "no_action"

def baseline_b1_dialog_threshold(p_model: float, cfg: EngineConfig) -> str:
    return "dialog" if p_model >= cfg.baseline_p_dialog else "no_action"

def baseline_b2_two_threshold(p_model: float, cfg: EngineConfig) -> str:
    if p_model >= cfg.baseline_p_action:
        return "action"
    return "dialog" if p_model >= cfg.baseline_p_dialog else "no_action"


# =============================
# Figures (Fig 1–7)
# =============================

def plot_fig7_dwell_time(cfg: EngineConfig, msg_lens: List[int], dwells: List[float], path: str) -> None:
    if not HAVE_MPL:
        return
    plt.figure()
    plt.scatter(msg_lens, dwells, s=10)
    xs = list(range(cfg.msg_len_min, cfg.msg_len_max + 1, 10))
    ys = []
    for x in xs:
        s = sigmoid((x - cfg.dwell_center_bytes) / cfg.dwell_width_bytes)
        y = cfg.dwell_min_sec + (cfg.dwell_max_sec - cfg.dwell_min_sec) * s
        ys.append(y)
    plt.plot(xs, ys)
    plt.xlabel("Length of message (bytes)")
    plt.ylabel("Dwell time before action (sec)")
    plt.title("Fig 7-like: Dwell time vs message length")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

def plot_fig4_threshold_action_noaction(utils: Dict[str, Tuple[float, float]], path: str) -> None:
    if not HAVE_MPL:
        return
    uNA_G, uNA_NG = utils["no_action"]
    uA_G, uA_NG = utils["action"]
    xs = [i / 200 for i in range(201)]
    eu_no = [p * uNA_G + (1 - p) * uNA_NG for p in xs]
    eu_act = [p * uA_G + (1 - p) * uA_NG for p in xs]
    pstar = threshold_p_star(uNA_G, uNA_NG, uA_G, uA_NG)

    plt.figure()
    plt.plot(xs, eu_no, label="No Action")
    plt.plot(xs, eu_act, label="Action")
    plt.axvline(pstar)
    plt.xlabel("p(G|E)")
    plt.ylabel("Expected utility")
    plt.title("Fig 4-like: Action vs No Action threshold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

def plot_fig5_context_shifted_thresholds(cfg: EngineConfig, util_model: UtilityModel, path: str) -> None:
    if not HAVE_MPL:
        return

    contexts = [
        ("low_nr, low_urg", 0.0, 0.2),
        ("low_nr, high_urg", 0.0, 0.9),
        ("high_nr, low_urg", 0.6, 0.2),
        ("high_nr, high_urg", 0.6, 0.9),
    ]

    xs = [i / 200 for i in range(201)]

    plt.figure(figsize=(10, 8))
    for idx, (name, nr, urg) in enumerate(contexts, start=1):
        utils = util_model.utilities(nr, urg)
        uNA_G, uNA_NG = utils["no_action"]
        uA_G, uA_NG = utils["action"]
        uD_G, uD_NG = utils["dialog"]

        eu_no = [p * uNA_G + (1 - p) * uNA_NG for p in xs]
        eu_act = [p * uA_G + (1 - p) * uA_NG for p in xs]
        eu_dlg = [p * uD_G + (1 - p) * uD_NG for p in xs]

        p_na_d = threshold_p_star(uNA_G, uNA_NG, uD_G, uD_NG)
        p_d_a = threshold_p_star(uD_G, uD_NG, uA_G, uA_NG)

        plt.subplot(2, 2, idx)
        plt.plot(xs, eu_no, label="No Action")
        plt.plot(xs, eu_dlg, label="Dialog")
        plt.plot(xs, eu_act, label="Action")
        plt.axvline(p_na_d)
        plt.axvline(p_d_a)
        plt.title(f"{name}\n(not_ready={nr:.1f}, urgency={urg:.1f})")
        plt.xlabel("p(G|E)")
        plt.ylabel("Expected utility")
        if idx == 1:
            plt.legend()

    plt.suptitle("Fig 5 analog: Context-shifted expected utilities and thresholds")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

def plot_fig6_thresholds(utils: Dict[str, Tuple[float, float]], path: str) -> None:
    if not HAVE_MPL:
        return
    uNA_G, uNA_NG = utils["no_action"]
    uA_G, uA_NG = utils["action"]
    uD_G, uD_NG = utils["dialog"]
    xs = [i / 200 for i in range(201)]
    eu_no = [p * uNA_G + (1 - p) * uNA_NG for p in xs]
    eu_act = [p * uA_G + (1 - p) * uA_NG for p in xs]
    eu_dlg = [p * uD_G + (1 - p) * uD_NG for p in xs]
    p_na_d = threshold_p_star(uNA_G, uNA_NG, uD_G, uD_NG)
    p_d_a = threshold_p_star(uD_G, uD_NG, uA_G, uA_NG)

    plt.figure()
    plt.plot(xs, eu_no, label="No Action")
    plt.plot(xs, eu_dlg, label="Dialog")
    plt.plot(xs, eu_act, label="Action")
    plt.axvline(p_na_d)
    plt.axvline(p_d_a)
    plt.xlabel("p(G|E)")
    plt.ylabel("Expected utility")
    plt.title("Fig 6-like: No Action vs Dialog vs Action thresholds")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

def plot_fig1_manual_invocation_summary(rows: List[Dict[str, Any]], path: str) -> None:
    if not HAVE_MPL:
        return
    manual = [r for r in rows if r["lookout_modality"] == "manual_invocation"]
    if not manual:
        return

    n = len(manual)
    click = sum(1 for r in manual if int(r.get("manual_click_invoke", 0)) == 1)
    hover_only = sum(1 for r in manual if int(r.get("manual_hover_inspect", 0)) == 1 and int(r.get("manual_click_invoke", 0)) == 0)
    no_action = sum(1 for r in manual if r.get("agent_action") == "no_action")
    other = n - (click + hover_only + no_action)

    cats = ["click_invoke", "hover_only", "no_action", "other"]
    vals = [click, hover_only, no_action, other]

    plt.figure()
    plt.bar(cats, vals)
    plt.ylabel("Count")
    plt.title("Fig 1 analog: Manual invocation outcomes")
    plt.xticks(rotation=20, ha="right")
    for i, v in enumerate(vals):
        plt.text(i, v, f"{v} ({(v/n)*100:.1f}%)", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

def plot_fig2_explicit_agent_flow(rows: List[Dict[str, Any]], path: str) -> None:
    if not HAVE_MPL:
        return
    exp = [r for r in rows if r["lookout_modality"] == "explicit_agent"]
    dialog_eps = [r for r in exp if r["agent_action"] == "dialog"]
    if not dialog_eps:
        return

    n = len(dialog_eps)
    accept = sum(1 for r in dialog_eps if r["user_response"] == "accept")
    reject = sum(1 for r in dialog_eps if r["user_response"] == "reject")
    ignore = sum(1 for r in dialog_eps if r["user_response"] == "ignore")
    refine = sum(1 for r in dialog_eps if r["user_response"] == "accept" and int(r.get("refine_after_accept", 0)) == 1)

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    cats = ["accept", "reject", "ignore"]
    vals = [accept, reject, ignore]
    plt.bar(cats, vals)
    plt.title("Explicit agent (dialog) responses")
    plt.ylabel("Count")
    for i, v in enumerate(vals):
        plt.text(i, v, f"{v} ({(v/n)*100:.1f}%)", ha="center", va="bottom")

    plt.subplot(1, 2, 2)
    cats2 = ["accept", "accept+refine"]
    vals2 = [accept, refine]
    plt.bar(cats2, vals2)
    plt.title("Refinement after accept")
    denom = accept if accept > 0 else 1
    plt.ylabel("Count")
    plt.text(0, accept, f"{accept}", ha="center", va="bottom")
    plt.text(1, refine, f"{refine} ({(refine/denom)*100:.1f}% of accepts)", ha="center", va="bottom")

    plt.suptitle("Fig 2 analog: Explicit agent dialog → confirmation → refinement")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

def plot_fig3_auto_scoping_policy_map(cfg: EngineConfig, rows: List[Dict[str, Any]], path: str) -> None:
    """
    Fig 3 analog: Dominant policy region over (p_model × not_ready).
    Legend is intentionally NOT drawn over the heatmap; the colorbar encodes the mapping.
    """
    if not HAVE_MPL:
        return

    auto = [r for r in rows if r["lookout_modality"] == "auto_scoping"]
    if len(auto) < 1000:
        auto = rows

    bins_p = cfg.fig3_bins_p
    bins_nr = cfg.fig3_bins_nr

    p_edges = [i / bins_p for i in range(bins_p + 1)]
    nr_edges = [i / bins_nr for i in range(bins_nr + 1)]

    action_to_code = {"no_action": 0, "dialog": 1, "scope": 2, "action": 3}
    grid = [[{} for _ in range(bins_p)] for __ in range(bins_nr)]

    def bin_index(val: float, edges: List[float]) -> int:
        if val <= edges[0]:
            return 0
        if val >= edges[-1]:
            return len(edges) - 2
        lo, hi = 0, len(edges) - 2
        while lo <= hi:
            mid = (lo + hi) // 2
            if edges[mid] <= val < edges[mid + 1]:
                return mid
            if val < edges[mid]:
                hi = mid - 1
            else:
                lo = mid + 1
        return len(edges) - 2

    for r in auto:
        p = float(r["p_model"])
        nr = float(r["not_ready_score"])
        code = action_to_code.get(r["agent_action"], 0)
        pi = bin_index(p, p_edges)
        ni = bin_index(nr, nr_edges)
        d = grid[ni][pi]
        d[code] = d.get(code, 0) + 1

    dom = [[0 for _ in range(bins_p)] for __ in range(bins_nr)]
    for ni in range(bins_nr):
        for pi in range(bins_p):
            counts = grid[ni][pi]
            if not counts:
                dom[ni][pi] = 0
                continue
            # tie-break preference: action > scope > dialog > no_action
            maxc = max(counts.values())
            for c in (3, 2, 1, 0):
                if counts.get(c, 0) == maxc:
                    dom[ni][pi] = c
                    break

    plt.figure(figsize=(9, 5))
    im = plt.imshow(dom, aspect="auto", origin="lower")
    cbar = plt.colorbar(im, ticks=[0, 1, 2, 3])
    cbar.set_label("Dominant action code\n0=no_action, 1=dialog, 2=scope, 3=action")

    plt.xlabel("p_model bin (low → high)")
    plt.ylabel("not_ready bin (low → high)")
    plt.title(
        "Fig 3 analog: Dominant policy region over (p_model × not_ready)\n"
        "Higher p_model favors intervention; higher not_ready favors deferral/scoping"
    )
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


# =============================
# Quality report
# =============================

def build_quality_report(cfg: EngineConfig, rows: List[Dict[str, Any]], msg_lens: List[int], dwells: List[float]) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    n = len(rows)
    if n == 0:
        return "Empty dataset.\n"

    p_true = [float(r["p_true"]) for r in rows]
    p_model = [float(r["p_model"]) for r in rows]
    y = [int(r["true_goal"]) for r in rows]
    nr = [float(r["not_ready_score"]) for r in rows]

    decisions: Dict[str, int] = {}
    responses: Dict[str, int] = {}
    modalities: Dict[str, int] = {}
    for r in rows:
        decisions[r["agent_action"]] = decisions.get(r["agent_action"], 0) + 1
        responses[r["user_response"]] = responses.get(r["user_response"], 0) + 1
        modalities[r["lookout_modality"]] = modalities.get(r["lookout_modality"], 0) + 1

    auroc_true = approx_auroc(p_true, y)
    auroc_model = approx_auroc(p_model, y)

    calib_model, ece_model = calibration_table_and_ece(p_model, y, cfg.calibration_bins)

    y_action = [1 if r["agent_action"] == "action" else 0 for r in rows]
    slope_p, z_p, pval_p = slope_test_binary_y(p_model, y_action)
    slope_nr, z_nr, pval_nr = slope_test_binary_y(nr, y_action)

    nr_sorted = sorted(nr)
    nr_p75 = quantile(nr_sorted, 0.75)

    # Fig 7 correlation
    dwell_mean = mean(dwells)
    len_mean = mean([float(x) for x in msg_lens])
    cov = mean([(dwells[i] - dwell_mean) * (msg_lens[i] - len_mean) for i in range(len(dwells))])
    varx = mean([(msg_lens[i] - len_mean) ** 2 for i in range(len(dwells))])
    vary = mean([(dwells[i] - dwell_mean) ** 2 for i in range(len(dwells))])
    corr = cov / math.sqrt(varx * vary) if varx > 0 and vary > 0 else float("nan")

    # Baseline comparisons (counterfactual expected utility)
    outcome = OutcomeModel(cfg)
    util_model = UtilityModel(cfg)
    baseline_specs = [
        ("B0 always no_action", baseline_b0_no_action),
        (f"B1 dialog if p≥{cfg.baseline_p_dialog:.2f}", baseline_b1_dialog_threshold),
        (f"B2 action if p≥{cfg.baseline_p_action:.2f} else dialog if p≥{cfg.baseline_p_dialog:.2f}", baseline_b2_two_threshold),
    ]
    deltas = {name: [] for name, _ in baseline_specs}
    for r in rows:
        tg = int(r["true_goal"])
        notr = float(r["not_ready_score"])
        urg = float(r.get("urgency_0_1", 0.5))
        utils = util_model.utilities(notr, urg)
        realized = float(r["realized_utility"])
        pm = float(r["p_model"])
        for name, fn in baseline_specs:
            a = fn(pm, cfg)
            beu = outcome.expected_utility_for_action(a, tg, notr, utils)
            deltas[name].append(realized - beu)

    # Fig 1/2 summary stats (for report)
    manual = [r for r in rows if r["lookout_modality"] == "manual_invocation"]
    manual_n = len(manual)
    manual_click = sum(1 for r in manual if int(r.get("manual_click_invoke", 0)) == 1)
    manual_hover_only = sum(1 for r in manual if int(r.get("manual_hover_inspect", 0)) == 1 and int(r.get("manual_click_invoke", 0)) == 0)
    manual_no_action = sum(1 for r in manual if r["agent_action"] == "no_action")

    exp = [r for r in rows if r["lookout_modality"] == "explicit_agent"]
    exp_dialog = [r for r in exp if r["agent_action"] == "dialog"]
    expd_n = len(exp_dialog)
    expd_accept = sum(1 for r in exp_dialog if r["user_response"] == "accept")
    expd_refine = sum(1 for r in exp_dialog if r["user_response"] == "accept" and int(r.get("refine_after_accept", 0)) == 1)

    def flag(pass_cond: bool, warn_cond: bool) -> str:
        if pass_cond:
            return "PASS"
        if warn_cond:
            return "WARN"
        return "FAIL"

    flag_ece_model = flag(ece_model <= 0.08, ece_model <= 0.15)
    flag_auroc_model = flag(0.80 <= auroc_model <= 0.995, 0.65 <= auroc_model < 0.80 or auroc_model > 0.995)
    flag_slope_p = flag(slope_p > 0 and pval_p < cfg.slope_alpha, slope_p > 0 and pval_p < cfg.slope_warn_alpha)
    flag_slope_nr = flag(slope_nr < 0 and pval_nr < cfg.slope_alpha, slope_nr < 0 and pval_nr < cfg.slope_warn_alpha)
    flag_not_ready = flag(nr_p75 >= 0.10, nr_p75 >= 0.04)

    mean_delta_b0 = mean(deltas[baseline_specs[0][0]])
    flag_net_b0 = flag(mean_delta_b0 > 0.00, mean_delta_b0 > -0.02)

    # Interpretability bins
    action_by_p = binned_rate_equal_count(rows, "p_model", lambda rr: rr["agent_action"] == "action",
                                         cfg.report_bins_target, cfg.report_min_bin_size)
    ignore_by_nr = binned_rate_not_ready_zero_collapse(rows, "not_ready_score", lambda rr: rr["user_response"] == "ignore",
                                                       cfg.report_bins_target, cfg.report_min_bin_size, cfg.not_ready_zero_epsilon)

    lines: List[str] = []
    lines.append("LookOut-inspired Mixed-Initiative Engine — Quality Report (v1.0)")
    lines.append("Developed by DBbun LLC — January 2026")
    lines.append(f"Generated: {now}")
    lines.append("")
    lines.append("Files")
    lines.append(f"- CSV:   {outpath(cfg,'csv')}")
    lines.append(f"- JSONL: {outpath(cfg,'jsonl')}")
    lines.append(f"- Config:{outpath(cfg,'config.json')}")
    lines.append(f"- Report:{outpath(cfg,'quality_report.txt')}")
    if cfg.make_plots_if_possible and HAVE_MPL:
        lines.append("- Figures:")
        lines.append("  • output/fig1_manual_invocation_summary.png")
        lines.append("  • output/fig2_explicit_agent_flow.png")
        lines.append("  • output/fig3_auto_scoping_policy_map.png")
        lines.append("  • output/fig4_threshold_action_vs_noaction.png")
        lines.append("  • output/fig5_context_shifted_thresholds.png")
        lines.append("  • output/fig6_thresholds_action_dialog_noaction.png")
        lines.append("  • output/fig7_dwell_time_vs_message_length.png")
    lines.append("")

    lines.append("1) Distributions")
    lines.append(f"- rows: {n:,}")
    lines.append("Modality distribution:")
    for k in sorted(modalities.keys()):
        lines.append(f"- {k}: {modalities[k]:,} ({fmt_pct(modalities[k]/n)})")
    lines.append("Action distribution:")
    for k in sorted(decisions.keys()):
        lines.append(f"- {k}: {decisions[k]:,} ({fmt_pct(decisions[k]/n)})")
    lines.append("Response distribution:")
    for k in sorted(responses.keys()):
        lines.append(f"- {k}: {responses[k]:,} ({fmt_pct(responses[k]/n)})")
    lines.append("")

    lines.append("2) Uncertainty quality")
    lines.append(f"- AUROC(p_true):  {auroc_true:.4f}  (calibrated-by-construction via latent evidence)")
    lines.append(f"- AUROC(p_model): {auroc_model:.4f}  => {flag_auroc_model}")
    lines.append(f"- ECE(p_model):   {ece_model:.4f}  => {flag_ece_model}")
    lines.append("")
    lines.append("Reliability (p_model): avg_pred -> empirical_rate (n)")
    for avg_pred, emp, bn in calib_model:
        lines.append(f"- {avg_pred:.3f} -> {emp:.3f} (n={bn})")
    lines.append("")

    lines.append("3) Sensitivity tests")
    lines.append("Action sensitivity to p_model (should be +):")
    lines.append(f"- slope: {slope_p:+.6f}, z: {z_p:+.3f}, p: {pval_p:.3e} => {flag_slope_p}")
    lines.append("Action sensitivity to not_ready (should be -):")
    lines.append(f"- slope: {slope_nr:+.6f}, z: {z_nr:+.3f}, p: {pval_nr:.3e} => {flag_slope_nr}")
    lines.append(f"- not_ready health (p75): {nr_p75:.3f} => {flag_not_ready}")
    lines.append("")

    lines.append("4) Fig 7 alignment (dwell time vs message length)")
    lines.append(f"- corr(message_length, dwell_time): {corr:.4f} (should be positive)")
    lines.append("")

    lines.append("5) Fig 1/2 behavioral summaries (data-driven)")
    if manual_n > 0:
        lines.append(f"- Fig 1 manual_invocation n={manual_n}: click={manual_click} ({fmt_pct(manual_click/manual_n)}), hover_only={manual_hover_only} ({fmt_pct(manual_hover_only/manual_n)}), no_action={manual_no_action} ({fmt_pct(manual_no_action/manual_n)})")
    if expd_n > 0:
        lines.append(f"- Fig 2 explicit_agent(dialog) n={expd_n}: accept={expd_accept} ({fmt_pct(expd_accept/expd_n)}), refine|accept={fmt_pct(safe_div(expd_refine, max(1, expd_accept)))}")
    lines.append("")

    lines.append("6) Binned curves (interpretability)")
    lines.append("P(action | p_model):")
    for c, rate, bn, vmin, vmax in action_by_p[:cfg.report_top_lines]:
        lines.append(f"- p≈{c:.3f}: {fmt_pct(rate)} (n={bn}, [{vmin:.3f},{vmax:.3f}])")
    lines.append("")
    lines.append("P(ignore | not_ready):")
    for c, rate, bn, vmin, vmax in ignore_by_nr[:cfg.report_top_lines]:
        lines.append(f"- nr≈{c:.3f}: {fmt_pct(rate)} (n={bn}, [{vmin:.3f},{vmax:.3f}])")
    lines.append("")

    lines.append("7) Net benefit vs baselines (counterfactual expected utility)")
    for name, _ in baseline_specs:
        xs = deltas[name]
        lines.append(
            f"{name}: mean Δ={mean(xs):+.4f}, median Δ={quantile(sorted(xs),0.50):+.4f}, "
            f"P(Δ>0)={fmt_pct(safe_div(sum(1 for v in xs if v>0), len(xs)))}"
        )
    lines.append(f"- Beats B0: mean Δ={mean_delta_b0:+.4f} => {flag_net_b0}")
    lines.append("")

    lines.append("8) Threshold provenance (Figs 4–6)")
    lines.append("- Per-episode thresholds p* are computed analytically from context-shifted utilities and stored in the dataset.")
    lines.append("")

    lines.append("9) Quick rubric")
    lines.append(f"- ECE(p_model): {flag_ece_model}")
    lines.append(f"- AUROC(p_model): {flag_auroc_model}")
    lines.append(f"- Sensitivity p_model: {flag_slope_p}")
    lines.append(f"- Sensitivity not_ready: {flag_slope_nr}")
    lines.append(f"- Beats B0: {flag_net_b0}")
    lines.append("")

    return "\n".join(lines)


# =============================
# Generation
# =============================

def generate(cfg: EngineConfig) -> Tuple[List[Dict[str, Any]], List[int], List[float]]:
    random.seed(cfg.seed)
    ensure_dir(cfg.output_dir)

    user_model = UserModel(cfg)
    attention = AttentionModel(cfg)
    obs = ObservationModel(cfg)
    util_model = UtilityModel(cfg)
    policy = LookOutPolicy(cfg)
    outcome = OutcomeModel(cfg)

    rows: List[Dict[str, Any]] = []
    msg_lens: List[int] = []
    dwells: List[float] = []

    for ep in range(cfg.n_episodes):
        user_id = user_model.sample_user()
        msg_len = random.randint(cfg.msg_len_min, cfg.msg_len_max)

        dwell = attention.dwell_time(msg_len)
        tsf = attention.time_since_focus(dwell)
        not_ready = attention.not_ready(dwell, tsf)

        evidence = user_model.sample_evidence()
        p_true = user_model.p_true_from_evidence(user_id, evidence)
        true_goal = user_model.sample_true_goal(p_true)

        p_model = obs.p_model(p_true)

        urgency = sample_time_urgency()
        utils = util_model.utilities(not_ready, urgency)

        modality = policy.choose_modality()
        chosen_action, eus, thr = policy.decide(p_model, utils, modality)

        # Fig 1 event flags (manual invocation)
        hover_inspect = 1 if modality == "manual_invocation" and chosen_action != "no_action" else 0
        click_invoke = 1 if modality == "manual_invocation" and chosen_action == "action" else 0

        # Sample outcome
        user_response, realized, refined = outcome.sample(chosen_action, true_goal, not_ready, utils)

        # Fig 2 dialog confirmation proxy (explicit_agent + dialog + accept)
        dialog_confirmed = 1 if (modality == "explicit_agent" and chosen_action == "dialog" and user_response == "accept") else 0

        # Realized utility noise
        realized = clamp01(realized + random.gauss(0.0, cfg.realized_utility_noise_std))

        baseline_no = utils["no_action"][0] if true_goal else utils["no_action"][1]

        rows.append({
            "episode_id": ep,
            "user_id": user_id,

            # Attention (Fig 7)
            "message_length_bytes": msg_len,
            "dwell_time_sec": round(dwell, 6),
            "time_since_focus_sec": round(tsf, 6),
            "not_ready_score": round(not_ready, 6),

            # Evidence + uncertainty
            "evidence_score": round(evidence, 6),
            "true_goal": true_goal,
            "p_true": round(p_true, 6),
            "p_model": round(p_model, 6),

            # Context
            "urgency_0_1": round(urgency, 6),

            # LookOut modality + action
            "lookout_modality": modality,
            "agent_action": chosen_action,

            # Fig 1 signals
            "manual_hover_inspect": hover_inspect,
            "manual_click_invoke": click_invoke,

            # Fig 2 signal
            "dialog_confirmed": dialog_confirmed,

            # Outcomes
            "user_response": user_response,
            "refine_after_accept": refined,
            "realized_utility": round(realized, 6),
            "u_baseline_no_action": round(baseline_no, 6),

            # Expected utilities (where allowed; missing values set to None)
            "eu_no_action": round(p_model * utils["no_action"][0] + (1 - p_model) * utils["no_action"][1], 6),
            "eu_action": round(eus.get("action", float("nan")), 6) if "action" in eus else None,
            "eu_dialog": round(eus.get("dialog", float("nan")), 6) if "dialog" in eus else None,
            "eu_scope": round(eus.get("scope", float("nan")), 6) if "scope" in eus else None,

            # Thresholds p* (Figs 4–6)
            "p_star_noaction_action": round(thr["p_star_noaction_action"], 6),
            "p_star_noaction_dialog": round(thr["p_star_noaction_dialog"], 6),
            "p_star_dialog_action": round(thr["p_star_dialog_action"], 6),
            "p_star_noaction_scope": round(thr["p_star_noaction_scope"], 6),
            "p_star_scope_action": round(thr["p_star_scope_action"], 6),
            "p_star_scope_dialog": round(thr["p_star_scope_dialog"], 6),

            # Store context-shifted utilities for auditability
            "u_no_action_goal": round(utils["no_action"][0], 6),
            "u_no_action_no_goal": round(utils["no_action"][1], 6),
            "u_action_goal": round(utils["action"][0], 6),
            "u_action_no_goal": round(utils["action"][1], 6),
            "u_dialog_goal": round(utils["dialog"][0], 6),
            "u_dialog_no_goal": round(utils["dialog"][1], 6),
            "u_scope_goal": round(utils["scope"][0], 6),
            "u_scope_no_goal": round(utils["scope"][1], 6),
        })

        msg_lens.append(msg_len)
        dwells.append(dwell)

    return rows, msg_lens, dwells


# =============================
# Writers
# =============================

def write_csv(rows: List[Dict[str, Any]], path: str) -> None:
    if not rows:
        raise ValueError("No rows to write.")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

def write_jsonl(rows: List[Dict[str, Any]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

def write_config(cfg: EngineConfig, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

def write_report(text: str, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


# =============================
# Main
# =============================

def main() -> None:
    cfg = EngineConfig()
    ensure_dir(cfg.output_dir)

    rows, msg_lens, dwells = generate(cfg)

    # stable outputs
    csv_path = outpath(cfg, "csv")
    jsonl_path = outpath(cfg, "jsonl")
    cfg_path = outpath(cfg, "config.json")
    rpt_path = outpath(cfg, "quality_report.txt")

    write_csv(rows, csv_path)
    write_jsonl(rows, jsonl_path)
    write_config(cfg, cfg_path)

    report = build_quality_report(cfg, rows, msg_lens, dwells)
    write_report(report, rpt_path)

    # Figures
    if cfg.make_plots_if_possible and HAVE_MPL:
        util_model = UtilityModel(cfg)

        # Fig 7
        fig7_path = os.path.join(cfg.output_dir, "fig7_dwell_time_vs_message_length.png")
        plot_fig7_dwell_time(cfg, msg_lens[:5000], dwells[:5000], fig7_path)

        # Fig 4 & Fig 6 (use representative mid-row utilities)
        mid_row = rows[len(rows) // 2]
        utils_snapshot = {
            "no_action": (float(mid_row["u_no_action_goal"]), float(mid_row["u_no_action_no_goal"])),
            "action": (float(mid_row["u_action_goal"]), float(mid_row["u_action_no_goal"])),
            "dialog": (float(mid_row["u_dialog_goal"]), float(mid_row["u_dialog_no_goal"])),
        }
        fig4_path = os.path.join(cfg.output_dir, "fig4_threshold_action_vs_noaction.png")
        plot_fig4_threshold_action_noaction(utils_snapshot, fig4_path)

        fig6_path = os.path.join(cfg.output_dir, "fig6_thresholds_action_dialog_noaction.png")
        plot_fig6_thresholds(utils_snapshot, fig6_path)

        # Fig 5 (context shifts)
        fig5_path = os.path.join(cfg.output_dir, "fig5_context_shifted_thresholds.png")
        plot_fig5_context_shifted_thresholds(cfg, util_model, fig5_path)

        # Fig 1–3 analogs (data-driven)
        fig1_path = os.path.join(cfg.output_dir, "fig1_manual_invocation_summary.png")
        plot_fig1_manual_invocation_summary(rows, fig1_path)

        fig2_path = os.path.join(cfg.output_dir, "fig2_explicit_agent_flow.png")
        plot_fig2_explicit_agent_flow(rows, fig2_path)

        fig3_path = os.path.join(cfg.output_dir, "fig3_auto_scoping_policy_map.png")
        plot_fig3_auto_scoping_policy_map(cfg, rows, fig3_path)

    print("Generation complete:")
    print(f"- {csv_path}")
    print(f"- {jsonl_path}")
    print(f"- {cfg_path}")
    print(f"- {rpt_path}")
    if cfg.make_plots_if_possible and HAVE_MPL:
        print("- output/fig1_manual_invocation_summary.png")
        print("- output/fig2_explicit_agent_flow.png")
        print("- output/fig3_auto_scoping_policy_map.png")
        print("- output/fig4_threshold_action_vs_noaction.png")
        print("- output/fig5_context_shifted_thresholds.png")
        print("- output/fig6_thresholds_action_dialog_noaction.png")
        print("- output/fig7_dwell_time_vs_message_length.png")
    elif cfg.make_plots_if_possible and not HAVE_MPL:
        print("- matplotlib not available; skipped figure generation.")


if __name__ == "__main__":
    main()
