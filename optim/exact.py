"""Solveur exact (PuLP + CBC) du même problème que le glouton, pour mesurer l'écart à l'optimum (#20).

Programme linéaire en nombres entiers (couverture maximale budgétée) :
    x_c ∈ {0, 1}  on ouvre le candidat c
    z_ck ∈ [0, 1] le carreau k est attribué au candidat c (k compte pour c : même règle que new_pop)
    max   somme f_c × w_k × z_ck                f_c = p_usage (Citiz) ou 1 ; w_k = pop × (1 + equity_weight × besoin)
    s.c.  z_ck <= x_c
          somme_{c de type T} z_ck <= 1   pour chaque type T et carreau k : un habitant compte une fois par type
          somme coût_c × x_c <= budget
Pour bus et Vélhop (f = 1), les z sont regroupés en une variable y_Tk <= somme des x_c qui couvrent k.

Pour Citiz, le solveur attribue chaque carreau à la station retenue de plus fort p_usage ; le glouton,
lui, l'attribue à la première station choisie (règle simple, sans retour en arrière). Pour un même plan,
la valeur du solveur est donc au moins celle du glouton : l'optimum calculé ici est une borne haute, et
l'écart publié est un majorant de l'écart réel.

PuLP >= 3 n'embarque plus CBC : installer `pulp==2.9.0` (CBC inclus) ou `highspy` (solveur HiGHS).
"""

from __future__ import annotations

import time
from collections import defaultdict

import numpy as np

from optim.plan import TYPES, Inputs, Problem, prepare


def solver(time_limit: float | None = None, msg: bool = False):
    import pulp

    avail = pulp.listSolvers(onlyAvailable=True)
    if "PULP_CBC_CMD" in avail:
        return pulp.PULP_CBC_CMD(msg=msg, timeLimit=time_limit)
    if "HiGHS" in avail:
        return pulp.HiGHS(msg=msg, timeLimit=time_limit)
    raise RuntimeError("aucun solveur PuLP disponible : pip install 'pulp==2.9.0' ou highspy")


def restrict(pb: Problem, keep: np.ndarray) -> Problem:
    """Sous-problème limité à certains candidats (indices)."""
    keep = np.asarray(keep)
    return Problem(cands=pb.cands.iloc[keep].reset_index(drop=True), cover=[pb.cover[i] for i in keep],
                   weight=pb.weight, pop=pb.pop, factor=pb.factor[keep], cost=pb.cost[keep],
                   type_code=pb.type_code[keep], types=pb.types)


def solve_problem(pb: Problem, budget_eur: float, time_limit: float | None = None, relax: bool = False) -> dict:
    """Résout le PLNE ; renvoie indices choisis, valeur, statut, temps.

    `relax=True` : relaxation linéaire (x continus), qui donne une borne haute de l'optimum.
    `status` : « optimal », « limite de temps » (meilleure solution trouvée, pas forcément optimale) ou autre.
    """
    import pulp

    t0 = time.perf_counter()
    m = pulp.LpProblem("couverture_budget", pulp.LpMaximize)
    n = len(pb.cost)
    x = [pulp.LpVariable(f"x{i}", 0, 1, cat="Continuous" if relax else "Binary") for i in range(n)]
    obj = []
    by_tk: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i in range(n):
        if pb.cost[i] > budget_eur:
            m += x[i] == 0
        for k in pb.cover[i]:
            by_tk[(int(pb.type_code[i]), int(k))].append(i)
    for (t, k), cs in by_tk.items():
        w = pb.weight[k]
        if w <= 0:
            continue
        if np.all(pb.factor[cs] == 1.0):
            y = pulp.LpVariable(f"y{t}_{k}", 0, 1)
            m += y <= pulp.lpSum(x[i] for i in cs)
            obj.append(w * y)
        else:
            zs = []
            for i in cs:
                z = pulp.LpVariable(f"z{i}_{k}", 0, 1)
                m += z <= x[i]
                zs.append(z)
                obj.append(pb.factor[i] * w * z)
            m += pulp.lpSum(zs) <= 1
    m += pulp.lpSum(obj)
    m += pulp.lpSum(pb.cost[i] * x[i] for i in range(n)) <= budget_eur
    m.solve(solver(time_limit))
    chosen = [i for i in range(n) if (x[i].value() or 0) > 0.5]
    return {"chosen": chosen, "value": float(pulp.value(m.objective) or 0.0),
            "status": {1: "optimal", 2: "limite de temps"}.get(m.sol_status, pulp.LpStatus[m.status]),
            "seconds": time.perf_counter() - t0,
            "n_candidates": n}


def best_assignment_value(pb: Problem, chosen) -> float:
    """Valeur d'un ensemble de candidats avec la règle du solveur (meilleur p_usage par carreau)."""
    best = np.zeros((len(pb.types), len(pb.weight)))
    for i in chosen:
        t, idx = pb.type_code[i], pb.cover[i]
        best[t, idx] = np.maximum(best[t, idx], pb.factor[i])
    return float((best * pb.weight).sum())


def exact_plan(inputs: Inputs, budget_eur: float, equity_weight: float = 0.0, types=TYPES,
               horizon_years: int = 5, min_p_usage: float = 0.0, top_n: int | None = None,
               time_limit: float | None = None, relax: bool = False) -> dict:
    """Optimum PuLP (ou borne haute si `relax`). `top_n` : ne garder que les N meilleurs candidats par gain seul / coût (taille du PLNE)."""
    pb = prepare(inputs, equity_weight, tuple(types), horizon_years, min_p_usage)
    if top_n is not None and top_n < len(pb.cost):
        ratio = np.array([pb.factor[i] * pb.weight[pb.cover[i]].sum() / pb.cost[i] for i in range(len(pb.cost))])
        pb = restrict(pb, np.sort(np.argsort(-ratio)[:top_n]))
    res = solve_problem(pb, budget_eur, time_limit, relax)
    res["candidate_ids"] = pb.cands.candidate_id.iloc[res["chosen"]].tolist()
    res["problem"] = pb
    return res
