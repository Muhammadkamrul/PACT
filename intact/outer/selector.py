"""
intact/outer/selector.py   --  INTACT v11 §8.1, §8.2, §8.3
==========================================================
THE EPOCH DECISION.

    S(tau) = argmax_S [  sum_{j in S} ( V_j + theta_D D_j )
                       - sum_{{j,k} subset S} H(j,k)  ]
    subject to C1, C2, C3.

THREE PIECES
    sum V_j       the good these claims do, ACROSS ALL TENANTS
    sum theta_D D_j   how OVERDUE each claim is -- this is what stops starvation
    - sum H       the damage from letting coupled claims act together

V_j  =  sum_i  w_i * beta_{j,i}                                  (v11 §8.1)
    One number: "on balance, across all tenants, is it good for this claim
    to act right now?"
    *** THIS LINE IS WHERE MULTI-TENANCY ENTERS. ***  In prior art the
    equivalent quantity is the change in ONE global throughput number.  Here
    it is a sum over OWNED, PRICED, PRIORITISED objectives -- which is why
    this framework can say "Telco-2's stream intent outranks the host's
    energy intent right now" and theirs cannot.

H(j,k) = q_hat_{jk} * sum_i w_i * max(-gamma_{jk,i}, 0)           (v11 §8.2)
    (A) q_hat : will BOTH actually fire this epoch?  An xApp that only
        writes when a buffer threshold is crossed cannot conflict while the
        buffer is low.
    (B) the measured interaction, priority-weighted.
    WHY THE PRODUCT MATTERS: a pure "these two COULD conflict" graph blocks
    controllers for conflicts that never happen.  Multiplying by q_hat keeps
    the structure loose when the network is healthy and tightens it only
    under stress.

theta_D is the single knob trading efficiency against fairness.
    large -> strict rotation.   small -> highest-value claims dominate.

COMPLEXITY (v11 §8.7)
    Claims writing the SAME knob mutually exclude, forming groups where at
    most one is chosen.  Without soft edges the answer is: take the highest-
    scoring claim per knob -- sort, O(J log J).  Soft edges make it
    non-trivial, but with J <= 30 exhaustive enumeration over the per-knob
    product is microseconds.  A greedy fallback with a (1 - 1/e) guarantee
    is provided for larger J.

THE ONE MODELLING RULE THAT KEEPS IT SMALL:
    shared-pool contention must be constraint C2, NEVER a harm edge.  As an
    edge, every claim conflicts with every other, valid sets collapse to
    singletons, and the scheduler degenerates to round-robin over one
    controller at a time.  This is the most likely way the design fails.
"""
from __future__ import annotations
import itertools, random
from typing import Callable, Dict, List, Optional, Set, Tuple
from ..types import Claim


def claim_values(claims, intents, w, beta) -> Dict[str, float]:
    """V_j = sum_i w_i * beta_{j,i}"""
    return {j: sum(w[i] * beta[i].get(j, 0.0) for i in intents) for j in claims}


def pair_harm(pairs, intents, w, gamma, q_hat) -> Dict[Tuple[str, str], float]:
    """H(j,k), for structurally coupled pairs only."""
    H = {}
    for pr in pairs:
        damage = sum(w[i] * max(-gamma[i].get(pr, 0.0), 0.0) for i in intents)
        H[pr] = q_hat.get(pr, 1.0) * damage
    return H


def score(S: Set[str], V: Dict[str, float], D: Dict[str, float],
          H: Dict[Tuple[str, str], float], theta_D: float) -> float:
    """The objective, exactly as written in v11 §8.3."""
    s = sum(V[j] + theta_D * D[j] for j in S)
    for (a, b), h in H.items():
        if a in S and b in S:
            s -= h
    return s


class Selector:
    def __init__(self, claims: Dict[str, Claim], mask, cfg, log):
        self.claims = claims
        self.mask = mask
        self.theta_D = cfg["outer"]["theta_D"]
        self.exhaustive_limit = cfg["outer"]["exhaustive_limit"]
        self.log = log
        # candidate space: at most one claim per knob, plus "none"
        self.knob_groups = mask.knob_groups

    # ------------------------------------------------------------------
    def _candidates(self):
        """
        Enumerate one choice per knob group (including 'skip this knob').
        This automatically satisfies C1, so the only remaining filter is C2.
        """
        opts = [[None] + js for js in self.knob_groups.values()]
        for combo in itertools.product(*opts):
            yield {j for j in combo if j is not None}

    def candidate_count(self) -> int:
        n = 1
        for js in self.knob_groups.values():
            n *= len(js) + 1
        return n

    def select_performance_frontier(
            self, V, D, H, performance_slack: float = 0.0,
            require_exact: bool = True,
            secondary_weight: float = 1.0) -> Tuple[Set[str], Dict[str, float]]:
        """Lexicographic C4 selection with an explicit efficiency budget.

        Stage 1 finds the admissible portfolio with the largest predicted
        weighted-margin improvement (``V - H``).  Stage 2 lets C4 deficit
        choose only among portfolios whose Stage-1 score is within
        ``performance_slack`` of that optimum.  This prevents a large virtual
        queue from turning a predicted harmful claim into a beneficial one.

        The old additive objective had no such bound: increasing theta_D was
        *supposed* to trade efficiency for service fairness, so the observed
        monotone loss of wIF was expected behavior rather than a numerical
        mystery.  Here the maximum predicted sacrifice is explicit and
        auditable on every epoch.
        """
        slack = max(0.0, float(performance_slack))
        n_cand = self.candidate_count()
        if n_cand > self.exhaustive_limit:
            if require_exact:
                raise RuntimeError(
                    f"performance-frontier selection needs {n_cand} candidates, "
                    f"above outer.exhaustive_limit={self.exhaustive_limit}; "
                    "raise the limit or reduce simultaneous claim groups. "
                    "No heuristic was used.")
            # Safe fallback: pure-performance local search.  It does not use
            # deficit because doing so without enumerating the frontier would
            # invalidate the performance-loss guarantee.
            zero = {j: 0.0 for j in self.claims}
            S, primary = self.select(V, zero, H)
            return S, {
                "candidate_count": float(n_cand), "frontier_size": 1.0,
                "best_primary": float(primary), "selected_primary": float(primary),
                "primary_loss": 0.0, "selected_deficit": 0.0,
                "exact": 0.0, "c4_changed_choice": 0.0,
            }

        zero = {j: 0.0 for j in self.claims}
        rows = []
        best_set, best_primary = set(), float("-inf")
        for S in self._candidates():
            if not self.mask.is_admissible(S):
                continue
            primary = score(S, V, zero, H, 0.0)
            rows.append((set(S), primary))
            # Strict comparison preserves the existing selector's
            # deterministic tie behavior (the first enumerated optimum).
            if primary > best_primary:
                best_set, best_primary = set(S), primary

        if not rows:
            raise RuntimeError("no admissible outer-loop portfolio")

        threshold = best_primary - slack
        frontier = [(S, p) for S, p in rows if p >= threshold - 1e-12]
        chosen, chosen_primary = set(best_set), best_primary
        chosen_secondary = sum(float(D.get(j, 0.0)) for j in chosen)
        best_utility = chosen_primary + float(secondary_weight) * chosen_secondary
        for S, primary in frontier:
            secondary = sum(float(D.get(j, 0.0)) for j in S)
            utility = primary + float(secondary_weight) * secondary
            # Keep the primary optimum on an exact tie.  This makes theta_D=0
            # exactly equivalent to the deficit-free ablation.
            if utility > best_utility + 1e-12:
                chosen, chosen_primary = set(S), primary
                chosen_secondary, best_utility = secondary, utility

        return chosen, {
            "candidate_count": float(n_cand),
            "frontier_size": float(len(frontier)),
            "best_primary": float(best_primary),
            "selected_primary": float(chosen_primary),
            "primary_loss": float(max(0.0, best_primary - chosen_primary)),
            "selected_deficit": float(chosen_secondary),
            "exact": 1.0,
            "c4_changed_choice": float(chosen != best_set),
        }

    def select_utility_frontier(
            self,
            primary_utility: Callable[[Set[str]], float],
            allowed: Optional[Set[str]] = None,
            secondary_utility: Optional[Callable[[Set[str]], float]] = None,
            performance_slack: float = 0.0,
            require_exact: bool = True) -> Tuple[Set[str], Dict[str, float]]:
        """Select an admissible set using a possibly non-additive utility.

        ``select`` and ``select_performance_frontier`` assume that every
        claim contributes a fixed additive value.  A fulfilment objective is
        not additive: moving an intent from margin -0.01 to +0.01 is valuable,
        while adding the same +0.02 to an already comfortable intent usually
        does not change fulfilment.  This method evaluates the complete set so
        the selector can optimise that threshold-aware objective exactly.

        The optional secondary objective is allowed to break near-ties only
        inside an explicit primary-performance slack.  This is the safe place
        for commercial priority or C3 recovery; C4 claim deficit is
        deliberately not implicit here.
        """
        slack = max(0.0, float(performance_slack))
        n_cand = self.candidate_count()
        if n_cand > self.exhaustive_limit and require_exact:
            raise RuntimeError(
                f"non-additive utility selection needs {n_cand} candidates, "
                f"above outer.exhaustive_limit={self.exhaustive_limit}; "
                "raise the limit or reduce simultaneous claim groups. "
                "No heuristic was used.")
        if n_cand > self.exhaustive_limit:
            raise RuntimeError(
                "a heuristic is not implemented for non-additive fulfilment "
                "utility; using one would change the registered objective")

        allowed = set(self.claims) if allowed is None else set(allowed)
        secondary_utility = secondary_utility or (lambda _S: 0.0)
        rows = []
        best_set, best_primary = set(), float("-inf")
        for S in self._candidates():
            if not S <= allowed or not self.mask.is_admissible(S):
                continue
            primary = float(primary_utility(set(S)))
            secondary = float(secondary_utility(set(S)))
            rows.append((set(S), primary, secondary))
            if primary > best_primary + 1e-12:
                best_set, best_primary = set(S), primary
        if not rows:
            raise RuntimeError("no admissible outer-loop portfolio")

        threshold = best_primary - slack
        frontier = [r for r in rows if r[1] >= threshold - 1e-12]
        chosen, chosen_primary = set(best_set), best_primary
        chosen_secondary = float(secondary_utility(chosen))
        for S, primary, secondary in frontier:
            # The secondary criterion is lexicographic, not an unbounded
            # additive reward.  It therefore cannot sacrifice more than the
            # declared primary slack.
            if secondary > chosen_secondary + 1e-12:
                chosen, chosen_primary = set(S), primary
                chosen_secondary = secondary

        return chosen, {
            "candidate_count": float(n_cand),
            "frontier_size": float(len(frontier)),
            "best_primary": float(best_primary),
            "selected_primary": float(chosen_primary),
            "primary_loss": float(max(0.0, best_primary - chosen_primary)),
            "selected_secondary": float(chosen_secondary),
            "selected_deficit": 0.0,
            "exact": 1.0,
            "c4_changed_choice": 0.0,
        }

    def select(self, V, D, H) -> Tuple[Set[str], float]:
        n_cand = self.candidate_count()

        if n_cand <= self.exhaustive_limit:
            best, best_s = set(), float("-inf")
            for S in self._candidates():
                if not self.mask.is_admissible(S):
                    continue
                sc = score(S, V, D, H, self.theta_D)
                if sc > best_s:
                    best, best_s = set(S), sc
            return best, best_s

        # ---- structured local search for large J ----------------------
        self.log.debug("candidate space %d > limit; local search", n_cand)
        # (a) seed: best claim per knob, ignoring C2 and H.  Exact when there
        #     are no soft edges and no C2 sets.
        S = set()
        for param, js in self.knob_groups.items():
            best_j = max(js, key=lambda j: V[j] + self.theta_D * D[j])
            if V[best_j] + self.theta_D * D[best_j] > 0:
                S.add(best_j)
        # (b) repair C2: drop the weakest member of each violated set
        for _ in range(len(self.claims)):
            bad = next((fs for fs in self.mask.c2_forbidden if fs <= S), None)
            if bad is None:
                break
            S.discard(min(bad, key=lambda j: V[j] + self.theta_D * D[j]))
        # (c) local search: drop / add / SWAP.  The swap move is what lets a
        #     starved claim displace an incumbent it is mutually exclusive
        #     with -- exactly what pure greedy could never do.
        improved = True
        while improved:
            improved = False
            cur = score(S, V, D, H, self.theta_D)
            for j in sorted(S):                       # drops
                T = S - {j}
                if self.mask.is_admissible(T) and score(T, V, D, H, self.theta_D) > cur + 1e-12:
                    S, improved = T, True
                    break
            if improved:
                continue
            for j in sorted(set(self.claims) - S):    # adds
                T = S | {j}
                if self.mask.is_admissible(T) and score(T, V, D, H, self.theta_D) > cur + 1e-12:
                    S, improved = T, True
                    break
            if improved:
                continue
            for j in sorted(S):                       # SWAPS
                for k in sorted(set(self.claims) - S):
                    T = (S - {j}) | {k}
                    if self.mask.is_admissible(T) and score(T, V, D, H, self.theta_D) > cur + 1e-12:
                        S, improved = T, True
                        break
                if improved:
                    break
        return S, score(S, V, D, H, self.theta_D)

    # ------------------------------------------------------------------
    def explore(self, S: Set[str], eps: float, at_risk: Set[str],
                claims_touching: Dict[str, Set[str]], rng: random.Random,
                counters: dict | None = None) -> Set[str]:
        """
        FORCED EXPLORATION -- v11 §7.3.  *** LOAD-BEARING. ***

        With probability eps, perturb the chosen set: drop one eligible
        claim or admit one shadowed claim.  This randomisation is the
        INSTRUMENT that makes beta an unbiased causal estimate.  Without it
        the difference in means is a correlation, and the whole attribution
        claim collapses.

        SAFETY VETO: never perturb a claim whose removal would touch an
        at-risk intent.  So exploration happens only when the system is
        healthy -- which is also when it is cheapest.
        """
        if rng.random() > eps:
            return S
        if counters is not None:
            counters["attempt"] += 1
        # sorted(): set iteration order depends on PYTHONHASHSEED and would
        # make rng.choice() non-reproducible across processes
        safe_to_drop = [j for j in sorted(S)
                        if not (claims_touching.get(j, set()) & at_risk)]
        safe_to_add = [j for j in sorted(self.claims) if j not in S
                       and not (claims_touching.get(j, set()) & at_risk)]
        if rng.random() < 0.5 and safe_to_drop:
            j = rng.choice(safe_to_drop); T = set(S); T.remove(j)
            if counters is not None: counters["forced_off"][j] += 1
        elif safe_to_add:
            j = rng.choice(safe_to_add); T = set(S) | {j}
            if counters is not None: counters["forced_on"][j] += 1
        else:
            if counters is not None: counters["veto"] += 1     # nothing safe
            return S
        if self.mask.is_admissible(T):
            if counters is not None: counters["accept"] += 1
            return T
        if counters is not None: counters["veto"] += 1
        return S
