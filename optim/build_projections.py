"""Construit data/ref/projections_omphale_dep.csv depuis les projections de population INSEE Omphale 2022.

    python optim/build_projections.py chemin/vers/projections_scenario_central.xlsx

Source : https://www.insee.fr/fr/statistiques/6652134 (scénario central, départements, 2018-2070).
Garde, par département : la population projetée chaque année (en milliers), l'âge moyen et la part des
75 ans et plus en 2018 et en 2070.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parent.parent / "data" / "ref" / "projections_omphale_dep.csv"
AGES_75P = ["[75,80)", "[80,85)", "[85,90)", "[90,95)", "[95,99]"]


def main(xlsx: str) -> None:
    book = pd.ExcelFile(xlsx)
    pop = book.parse("Population_DEP", header=5, dtype={"DEP": str}).dropna(subset=["DEP"])
    age = book.parse("Structure_Age_DEP", header=5, dtype={"DEP": str}).dropna(subset=["DEP"])
    out = pop[["DEP", "LIBDEP"] + [c for c in pop.columns if str(c).startswith("POP_")]].copy()
    for year in (2018, 2070):
        out[f"age_moyen_{year}"] = out.DEP.map(age.set_index("DEP")[f"AGE_MOY_{year}"])
        out[f"part_75p_{year}"] = out.DEP.map(age.set_index("DEP")[[f"{a}_{year}" for a in AGES_75P]].sum(axis=1) / 100)
    out.to_csv(OUT, index=False)
    print(f"{OUT} : {len(out)} départements")


if __name__ == "__main__":
    main(sys.argv[1])
