"""Projections de population INSEE Omphale 2022 (scénario central), par département.

Les carreaux Filosofi décrivent la population de 2021. Ces projections servent à dire combien d'habitants un
plan desservira à son horizon (`facteur_population`) et à donner le contexte démographique (`contexte`).
Elles sont départementales : le même facteur s'applique à toute l'Eurométropole, il ne change donc pas
l'ordre des emplacements (limite écrite dans docs/LIMITES.md).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

CSV = Path(__file__).resolve().parent.parent / "data" / "ref" / "projections_omphale_dep.csv"
ANNEE_FILOSOFI = 2021
ANNEE_DEBUT_PLAN = 2026


@lru_cache(maxsize=1)
def _table() -> pd.DataFrame:
    return pd.read_csv(CSV, dtype={"DEP": str}).set_index("DEP")


def facteur_population(dep: str, annee: int, base: int = ANNEE_FILOSOFI) -> float:
    """Population projetée du département en `annee`, rapportée à celle de `base` (1,0 si hors table)."""
    t = _table()
    annee = min(max(annee, 2018), 2070)
    if dep not in t.index or f"POP_{annee}" not in t:
        return 1.0
    return float(t.at[dep, f"POP_{annee}"] / t.at[dep, f"POP_{base}"])


def contexte(dep: str, horizon_years: int = 5) -> dict:
    """Chiffres de contexte pour l'application et les slides (rien au carreau)."""
    t = _table()
    if dep not in t.index:
        return {}
    r = t.loc[dep]
    fin = ANNEE_DEBUT_PLAN + horizon_years
    return {
        "departement": r.LIBDEP,
        "annee_horizon": fin,
        "evolution_horizon": facteur_population(dep, fin) - 1,
        "evolution_2040": facteur_population(dep, 2040) - 1,
        "part_75p_2018": float(r.part_75p_2018),
        "part_75p_2070": float(r.part_75p_2070),
        "age_moyen_2018": float(r.age_moyen_2018),
        "age_moyen_2070": float(r.age_moyen_2070),
    }
