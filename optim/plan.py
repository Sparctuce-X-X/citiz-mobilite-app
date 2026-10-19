"""Optimiseur sous budget : liste priorisée d'emplacements (algorithme glouton paresseux).

Problème : couverture maximale budgétée. Chaque candidat couvre des carreaux de 200 m (`covered_carreaux`,
400 m à pied). Un carreau ne compte pour un candidat de type T que s'il n'a aujourd'hui aucune option de
type T à 5 min (même règle que `new_pop`, voir fabric/nb_gold_candidates.ipynb ; bus : ni bus ni tram),
et seulement s'il n'est pas déjà couvert par un candidat du même type choisi plus tôt : un habitant n'est
jamais compté deux fois pour un même type.

    from optim.plan import load_inputs, greedy_plan, summarize
    inputs = load_inputs("snapshot")             # data/snapshot/*.parquet (optim/export_snapshot.py)
    inputs = load_inputs("snapshot", "data/snapshot/nancy")   # une autre ville (export_snapshot.py --ville nancy)
    plan = greedy_plan(inputs, budget_eur=500_000, equity_weight=0.5)
    print(summarize(plan, inputs))

Gain d'un candidat c de type T, compte tenu des choix déjà faits :
    gain(c) = facteur(c) × somme, sur les carreaux k nouveaux pour T, de pop_k × (1 + equity_weight × besoin_k)
avec facteur = p_usage pour Citiz (usage attendu, modèle #13), 1 pour bus et Vélhop.
Coût : capex + horizon_years × opex (data/ref/couts.csv ou table gold_costs).
À chaque tour on prend le meilleur rapport gain / coût qui tient dans le budget restant. Les gains ne font
que baisser quand le plan grandit : on ne recalcule que le candidat en tête de file (glouton paresseux).

besoin_k ∈ [0, 1] = moyenne de deux termes, pour que le curseur équité favorise les habitants qui en ont
le plus besoin :
  - accessibilité : (5 - classe) / 4, classe de gold_carreau_access (1 = aucune option à 5 min → 1 ; 5 → 0) ;
  - pauvreté : rang centile (0 à 1) de la part de ménages pauvres de l'IRIS du carreau
    (somme de menages_pauvres / somme de menages sur l'IRIS : jamais une valeur de carreau isolé,
    voir docs/ETHIQUE.md).
"""

from __future__ import annotations

import hashlib
import heapq
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT = ROOT / "data" / "snapshot"
COSTS_CSV = ROOT / "data" / "ref" / "couts.csv"
TYPES = ("citiz", "bus", "velhop")
CO2_T_PAR_VOITURE_AN = 2.6  # docs/IMPACT.md, borne basse ADEME
VOITURES_PAR_STATION_CITIZ = 1  # docs/IMPACT.md, hypothèse prudente

PLAN_COLUMNS = ["run_id", "rank", "candidate_id", "type", "lat", "lon", "cost_eur", "new_pop",
                "weighted_gain", "cumul_cost_eur", "cumul_pop", "budget_eur", "equity_weight"]


@dataclass
class Inputs:
    """Entrées de l'optimiseur.

    candidates : gold_candidates + p_usage (NaN hors Citiz).
    carreaux : gold_carreau_access + `besoin` (0 à 1) + `part_pauvres_iris`, `niveau_vie_somme`,
               `qpv` (si gold_carreau_qpv existe).
    costs : type, capex_eur, opex_annuel_eur, source.
    """

    candidates: pd.DataFrame
    carreaux: pd.DataFrame
    costs: pd.DataFrame


# ---------------------------------------------------------------- chargement

def snapshot_dir(directory: str | Path | None = None) -> Path:
    """Dossier du snapshot : argument, sinon variable d'environnement SNAPSHOT_DIR, sinon data/snapshot (Strasbourg).

    Une autre ville : data/snapshot/<ville>/ (python optim/export_snapshot.py --ville nancy)."""
    return Path(directory or os.environ.get("SNAPSHOT_DIR") or SNAPSHOT)


def _read_snapshot(name: str, directory: str | Path | None = None) -> pd.DataFrame | None:
    path = snapshot_dir(directory) / f"{name}.parquet"
    return pd.read_parquet(path) if path.exists() else None


def _read_onelake(name: str, schema: str = "dbo") -> pd.DataFrame | None:
    from deltalake import DeltaTable

    sys.path.insert(0, str(ROOT / "fabric"))
    import onelake_sync as o

    tok = os.environ.get("ONELAKE_TOKEN") or o.credential().get_token("https://storage.azure.com/.default").token
    base = o.tables_uri(schema)
    try:
        return DeltaTable(base + name, storage_options={"bearer_token": tok, "use_fabric_endpoint": "true"}).to_pandas()
    except Exception:  # noqa: BLE001  table absente
        return None


def build_inputs(candidates: pd.DataFrame, access: pd.DataFrame, costs: pd.DataFrame,
                 predictions: pd.DataFrame | None = None, carreaux_insee: pd.DataFrame | None = None,
                 qpv: pd.DataFrame | None = None) -> Inputs:
    """Assemble les entrées à partir des tables (utile aussi pour les tests sur un jeu fictif)."""
    cands = candidates.copy()
    if "p_usage" not in cands:
        if predictions is not None:
            cands = cands.merge(predictions[["candidate_id", "p_usage"]], on="candidate_id", how="left")
        else:
            cands["p_usage"] = np.nan
    car = access.copy()
    acces = ((5 - car["classe"].clip(1, 5)) / 4).astype(float)
    if carreaux_insee is not None and {"iris_code", "menages", "menages_pauvres"} <= set(carreaux_insee):
        ins = carreaux_insee.set_index("carreau_id")
        iris = ins.groupby("iris_code")[["menages", "menages_pauvres"]].sum()
        part = (iris.menages_pauvres / iris.menages.where(iris.menages > 0)).fillna(0)
        car["part_pauvres_iris"] = car.carreau_id.map(ins.iris_code).map(part).fillna(0).to_numpy()
        car["niveau_vie_somme"] = car.carreau_id.map(ins.niveau_vie_somme).to_numpy() \
            if "niveau_vie_somme" in ins else np.nan
        pauvrete = car["part_pauvres_iris"].rank(pct=True, method="average")
        car["besoin"] = (0.5 * acces + 0.5 * pauvrete).clip(0, 1)
    else:
        # sans menages_pauvres, le curseur équité ne voit plus la pauvreté : on le dit au lieu de l'ignorer
        warnings.warn("silver_carreaux sans iris_code / menages / menages_pauvres : besoin = accès seul "
                      "(relancer optim/export_snapshot.py)", stacklevel=2)
        car["besoin"] = acces
    if qpv is not None and "qpv" in qpv:
        car["qpv"] = car.carreau_id.map(qpv.set_index("carreau_id")["qpv"]).fillna(False).astype(bool)
    return Inputs(candidates=cands, carreaux=car, costs=costs.copy())


def load_inputs(source: str = "snapshot", directory: str | Path | None = None, schema: str = "dbo") -> Inputs:
    """Charge les entrées depuis le snapshot ("snapshot") ou le Lakehouse ("onelake").

    Snapshot : `directory`, sinon SNAPSHOT_DIR, sinon data/snapshot/ (Strasbourg) ; une autre ville :
    data/snapshot/<ville>/. Lakehouse : tables du schéma `schema` (Strasbourg : dbo ; Nancy : nancy).
    Coûts : table gold_costs si elle existe, sinon data/ref/couts.csv.
    """
    if source == "snapshot":
        def read(name):
            return _read_snapshot(name, directory)
    elif source == "onelake":
        def read(name):
            return _read_onelake(name, schema)
    else:
        raise KeyError(source)
    cands, access = read("gold_candidates"), read("gold_carreau_access")
    if cands is None or access is None:
        raise FileNotFoundError(f"gold_candidates / gold_carreau_access introuvables ({source}) : "
                                "lancer python optim/export_snapshot.py")
    costs = read("gold_costs")
    if costs is None:
        costs = pd.read_csv(COSTS_CSV)
    return build_inputs(cands, access, costs, predictions=read("gold_predictions"),
                        carreaux_insee=read("silver_carreaux"), qpv=read("gold_carreau_qpv"))


# ---------------------------------------------------------------- préparation commune (glouton et PuLP)

@dataclass
class Problem:
    """Problème indexé : candidats retenus, carreaux éligibles par candidat, poids, coûts."""

    cands: pd.DataFrame          # candidats retenus (filtrés), index 0..n-1
    cover: list[np.ndarray]      # indices des carreaux qui comptent pour chaque candidat
    weight: np.ndarray           # poids d'un carreau = pop × (1 + equity_weight × besoin)
    pop: np.ndarray
    factor: np.ndarray           # p_usage (Citiz) ou 1
    cost: np.ndarray
    type_code: np.ndarray        # 0..len(types)-1
    types: tuple[str, ...]


def prepare(inputs: Inputs, equity_weight: float = 0.0, types=TYPES, horizon_years: int = 5,
            min_p_usage: float = 0.0) -> Problem:
    car = inputs.carreaux.reset_index(drop=True)
    pos = pd.Series(np.arange(len(car)), index=car.carreau_id)
    eligible = {
        "citiz": (car.n_citiz == 0).to_numpy(),
        "velhop": (car.n_velhop == 0).to_numpy(),
        "bus": ((car.n_bus == 0) & (car.n_tram == 0)).to_numpy(),
    }
    costs = inputs.costs.set_index("type")
    unit_cost = {t: float(costs.loc[t, "capex_eur"] + horizon_years * costs.loc[t, "opex_annuel_eur"])
                 for t in costs.index}

    c = inputs.candidates[inputs.candidates["type"].isin(types)].copy()
    p = c["p_usage"].astype(float)
    is_citiz = c["type"].eq("citiz")
    c = c[~is_citiz | (p.fillna(0) >= min_p_usage)]
    c = c.reset_index(drop=True)
    factor = np.where(c["type"].eq("citiz"), c["p_usage"].astype(float).fillna(0), 1.0)

    cover = []
    for t, cov in zip(c["type"], c["covered_carreaux"]):
        idx = pos.reindex(list(cov)).dropna().to_numpy(dtype=np.int64)
        cover.append(idx[eligible[t][idx]])
    weight = car["pop"].to_numpy(float) * (1.0 + equity_weight * car["besoin"].to_numpy(float))
    tcode = {t: i for i, t in enumerate(types)}
    return Problem(cands=c, cover=cover, weight=weight, pop=car["pop"].to_numpy(float), factor=factor,
                   cost=c["type"].map(unit_cost).to_numpy(float),
                   type_code=c["type"].map(tcode).to_numpy(np.int64), types=tuple(types))


def run_id(budget_eur: float, equity_weight: float, types, horizon_years: int, min_p_usage: float) -> str:
    key = f"{budget_eur:.0f}|{equity_weight:g}|{','.join(sorted(types))}|{horizon_years}|{min_p_usage:g}"
    return "plan_" + hashlib.sha1(key.encode()).hexdigest()[:10]


# ---------------------------------------------------------------- glouton

def _greedy(pb: Problem, budget: float) -> list[int]:
    covered = np.zeros((len(pb.types), len(pb.weight)), dtype=bool)

    def gain(i: int) -> float:
        idx = pb.cover[i]
        return float(pb.factor[i] * pb.weight[idx[~covered[pb.type_code[i], idx]]].sum())

    heap = [(-gain(i) / pb.cost[i], i) for i in range(len(pb.cost)) if pb.cost[i] <= budget]
    heap = [h for h in heap if h[0] < 0]
    heapq.heapify(heap)
    chosen, left = [], budget
    while heap:
        neg_ratio, i = heapq.heappop(heap)
        if pb.cost[i] > left:  # le budget ne fait que baisser : on l'écarte définitivement
            continue
        g = gain(i)
        if g <= 0:
            continue
        ratio = g / pb.cost[i]
        if heap and ratio < -heap[0][0] - 1e-12:  # n'est plus en tête : on le remet avec son vrai gain
            heapq.heappush(heap, (-ratio, i))
            continue
        chosen.append(i)
        left -= pb.cost[i]
        covered[pb.type_code[i], pb.cover[i]] = True
    return chosen


def plan_value(pb: Problem, chosen: list[int]) -> float:
    """Gain pondéré total d'une liste ordonnée (même règle que le glouton)."""
    covered = np.zeros((len(pb.types), len(pb.weight)), dtype=bool)
    total = 0.0
    for i in chosen:
        idx = pb.cover[i]
        new = idx[~covered[pb.type_code[i], idx]]
        total += pb.factor[i] * pb.weight[new].sum()
        covered[pb.type_code[i], new] = True
    return total


def to_plan(pb: Problem, chosen: list[int], budget_eur: float, equity_weight: float, rid: str) -> pd.DataFrame:
    covered = np.zeros((len(pb.types), len(pb.weight)), dtype=bool)
    rows = []
    for rank, i in enumerate(chosen, start=1):
        idx = pb.cover[i]
        new = idx[~covered[pb.type_code[i], idx]]
        covered[pb.type_code[i], new] = True
        c = pb.cands.iloc[i]
        rows.append({"run_id": rid, "rank": rank, "candidate_id": c.candidate_id, "type": c["type"],
                     "lat": float(c.lat), "lon": float(c.lon), "cost_eur": float(pb.cost[i]),
                     "new_pop": float(pb.pop[new].sum()),
                     "weighted_gain": float(pb.factor[i] * pb.weight[new].sum())})
    plan = pd.DataFrame(rows, columns=PLAN_COLUMNS[:9])
    plan["cumul_cost_eur"] = plan["cost_eur"].cumsum()
    plan["cumul_pop"] = plan["new_pop"].cumsum()
    plan["budget_eur"] = float(budget_eur)
    plan["equity_weight"] = float(equity_weight)
    return plan[PLAN_COLUMNS]


def greedy_plan(inputs: Inputs, budget_eur: float, equity_weight: float = 0.0, types=TYPES,
                horizon_years: int = 5, min_p_usage: float = 0.0) -> pd.DataFrame:
    """Liste priorisée d'emplacements sous budget (colonnes de gold_plan).

    `new_pop` = habitants gagnés marginaux (non pondérés, non multipliés par p_usage) ;
    `weighted_gain` = gain utilisé pour classer (pondéré équité, × p_usage pour Citiz).
    Garde-fou classique : si le meilleur candidat seul vaut plus que tout le plan glouton, on le renvoie
    (garantie d'au moins 63 % de l'optimum).
    """
    types = tuple(types)
    pb = prepare(inputs, equity_weight, types, horizon_years, min_p_usage)
    chosen = _greedy(pb, budget_eur)
    singles = [(pb.factor[i] * pb.weight[pb.cover[i]].sum(), i) for i in range(len(pb.cost)) if pb.cost[i] <= budget_eur]
    if singles:
        best_val, best_i = max(singles)
        if best_val > plan_value(pb, chosen):
            chosen = [best_i]
    rid = run_id(budget_eur, equity_weight, types, horizon_years, min_p_usage)
    return to_plan(pb, chosen, budget_eur, equity_weight, rid)


# ---------------------------------------------------------------- résumé pour l'application

def gained_carreaux(plan: pd.DataFrame, inputs: Inputs) -> pd.DataFrame:
    """Carreaux nouvellement couverts par le plan (candidate_id, type, carreau_id, pop), dans l'ordre du plan.

    Usage interne / agrégats seulement : ne jamais afficher une ligne de carreau (docs/ETHIQUE.md).
    """
    car = inputs.carreaux.set_index("carreau_id")
    elig = {"citiz": car.n_citiz == 0, "velhop": car.n_velhop == 0, "bus": (car.n_bus == 0) & (car.n_tram == 0)}
    cov = inputs.candidates.set_index("candidate_id")["covered_carreaux"]
    seen: dict[str, set] = {t: set() for t in elig}
    rows = []
    for r in plan.itertuples():
        for k in cov[r.candidate_id]:
            if k in car.index and elig[r.type][k] and k not in seen[r.type]:
                seen[r.type].add(k)
                rows.append((r.candidate_id, r.type, k, float(car.at[k, "pop"])))
    return pd.DataFrame(rows, columns=["candidate_id", "type", "carreau_id", "pop"])


def summarize(plan: pd.DataFrame, inputs: Inputs) -> dict:
    """Indicateurs agrégés d'un plan (pour l'application et le pitch)."""
    n_type = {t: int((plan["type"] == t).sum()) for t in TYPES}
    budget = float(plan["budget_eur"].iloc[0]) if len(plan) else 0.0
    cout = float(plan["cost_eur"].sum())
    g = gained_carreaux(plan, inputs)
    car = inputs.carreaux.set_index("carreau_id")
    hab = float(g["pop"].sum())
    out = {
        "habitants_gagnes": int(round(hab)),  # un habitant peut compter pour deux types (bus et Citiz)
        "habitants_gagnes_distincts": int(round(car.loc[g.carreau_id.unique(), "pop"].sum())) if len(g) else 0,
        "habitants_gagnes_par_type": {t: int(round(g.loc[g["type"] == t, "pop"].sum())) for t in TYPES},
        "cout_total_eur": round(cout),
        "budget_eur": round(budget),
        "budget_restant_eur": round(budget - cout),
        "nb_emplacements": int(len(plan)),
        "nb_par_type": n_type,
        "cout_par_habitant_eur": round(cout / hab) if hab else None,
        "part_classe_1_2": round(float(g["pop"][g.carreau_id.map(car["classe"]).le(2).to_numpy()].sum() / hab), 3) if hab else None,
        "voitures_citiz_ajoutees": n_type["citiz"] * VOITURES_PAR_STATION_CITIZ,
        "co2_evite_t_an": int(round(n_type["citiz"] * VOITURES_PAR_STATION_CITIZ * CO2_T_PAR_VOITURE_AN)),
    }
    if "qpv" in car and hab:
        out["part_qpv"] = round(float(g["pop"][g.carreau_id.map(car["qpv"]).fillna(False).astype(bool).to_numpy()].sum() / hab), 3)
    if "niveau_vie_somme" in car and hab:
        nv = g.carreau_id.map(car["niveau_vie_somme"]).to_numpy(float)
        out["niveau_vie_moyen_gagnes_eur"] = int(round(np.nansum(nv) / hab))
        tot = car["niveau_vie_somme"].sum() / car["pop"].sum()
        out["niveau_vie_moyen_eurometropole_eur"] = int(round(tot))
    return out
