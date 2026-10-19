"""Chargement du snapshot et calcul des indicateurs de l'application (sans Streamlit, testable avec pytest).

Règles éthiques appliquées ici (docs/ETHIQUE.md) :
- carreaux `i_est_200 = 1` : jamais de valeur affichée au carreau (ils restent dans les totaux agrégés) ;
- indicateurs sociaux (niveau de vie, QPV) : seulement agrégés sur un groupe de carreaux assez grand ;
- CO₂ : 2,6 t CO₂e / an par voiture Citiz ajoutée (borne basse ADEME, docs/IMPACT.md), arrondi à l'unité,
  aucun CO₂ pour les arrêts de bus et stations vélo.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT_DIR = Path(os.environ.get("SNAPSHOT_DIR", ROOT / "data" / "snapshot"))

TABLES = (
    "gold_carreau_access",
    "gold_station_population",
    "gold_candidates",
    "gold_predictions",
    "gold_costs",
    "gold_carreau_qpv",
    "silver_carreaux",
)

MODES = ("tram", "bus", "citiz", "velhop")
MODE_LABELS = {"tram": "Tram", "bus": "Bus", "citiz": "Citiz (autopartage)", "velhop": "Vélhop (vélo)"}
TYPE_LABELS = {"citiz": "Station Citiz", "bus": "Arrêt de bus", "velhop": "Station Vélhop", "tram": "Arrêt de tram"}

# Échelle froid -> chaud de l'objectif 1 : elle décrit une quantité d'options, pas un jugement.
CLASS_COLORS = {1: "#313695", 2: "#74add1", 3: "#fee090", 4: "#f46d43", 5: "#a50026"}
CLASS_LABELS = {
    1: "1 · Aucune option à 5 min",
    2: "2 · Offre limitée",
    3: "3 · Offre moyenne",
    4: "4 · Bonne offre",
    5: "5 · Très bonne offre",
}
PROTECTED_COLOR = "#bdbdbd"
PROTECTED_LABEL = "Carreau protégé (secret statistique INSEE)"

TYPE_SINGULIERS = {"citiz": "station Citiz", "bus": "arrêt de bus", "velhop": "station Vélhop", "tram": "arrêt de tram"}
TYPE_PLURIELS = {"citiz": "stations Citiz", "bus": "arrêts de bus", "velhop": "stations Vélhop", "tram": "arrêts de tram"}

# Couleurs des équipements : distinctes de l'échelle des classes (classe 4 = #f46d43), et une forme par type.
TYPE_COLORS = {"citiz": "#2E7D4F", "bus": "#A3238E", "velhop": "#6A3FB5", "tram": "#5C677D"}
TYPE_SHAPES = {"citiz": "carré", "bus": "rond", "velhop": "losange", "tram": "triangle"}

# Score du modèle prédictif (gold_predictions.p_usage) : ressemblance aux stations actuelles, pas une
# probabilité de réservation.
P_USAGE_LABEL = "Ressemblance aux stations Citiz actuelles"
P_USAGE_HELP = ("Score du modèle prédictif (0 à 1) : à quel point l'emplacement ressemble aux lieux où Citiz est "
                "déjà implanté. Ce n'est pas une probabilité de réservation.")

CO2_T_PER_CITIZ_CAR = 2.6  # t CO₂e / an, docs/IMPACT.md
MIN_CARREAUX_AGREGAT = 5  # en dessous, on n'affiche pas d'indicateur social (risque de ré-identification)
MIN_MENAGES_AGREGAT = 11  # seuil du secret statistique INSEE


def hex_to_rgb(h: str, alpha: int = 255) -> list[int]:
    h = h.lstrip("#")
    return [int(h[i : i + 2], 16) for i in (0, 2, 4)] + [alpha]


# --------------------------------------------------------------------------- chargement


def nommer_velo(nom: str) -> None:
    """Libellés du vélo en libre-service de la ville affichée (Vélhop à Strasbourg, VélOStan'lib à Nancy)."""
    if nom == VELO_GENERIQUE:  # territoire sans vélo en libre-service : libellés génériques
        MODE_LABELS["velhop"] = "Vélo en libre-service"
        TYPE_LABELS["velhop"] = "Station de vélo en libre-service"
        TYPE_SINGULIERS["velhop"] = "station de vélo en libre-service"
        TYPE_PLURIELS["velhop"] = "stations de vélo en libre-service"
        return
    MODE_LABELS["velhop"] = f"{nom} (vélo)"
    TYPE_LABELS["velhop"] = f"Station {nom}"
    TYPE_SINGULIERS["velhop"] = f"station {nom}"
    TYPE_PLURIELS["velhop"] = f"stations {nom}"


VELO_GENERIQUE = "vélo en libre-service"


def load_city(directory: Path | str = SNAPSHOT_DIR) -> dict:
    """Noms affichés : `ville.json` du snapshot (écrit par export_snapshot.py --ville, ou par la chaîne nationale pour
    un EPCI), sinon Strasbourg. `est_epci` : snapshot d'un EPCI (data/snapshot/epci/<code>/), titré par son nom.
    Sans vélo en libre-service (`velo_nom` vide) : libellé générique, `a_velo` faux."""
    path = Path(directory) / "ville.json"
    if path.exists():
        city = {"ville": "strasbourg", "nom": "Eurométropole de Strasbourg", "velo_nom": "Vélhop"}
        city.update(json.loads(path.read_text(encoding="utf-8")))
    else:
        city = {"ville": "strasbourg", "nom": "Eurométropole de Strasbourg", "velo_nom": "Vélhop", "epci": "246700488"}
    city["est_epci"] = Path(directory).parent.name == "epci"
    city["a_velo"] = bool(city.get("velo_nom"))
    city["velo_nom"] = city.get("velo_nom") or VELO_GENERIQUE
    city["ville"] = str(city.get("ville") or city["nom"])
    if city.get("titre"):
        pass
    elif city["est_epci"] or city["ville"].isdigit():
        city["titre"] = city["nom"]
    else:
        city["titre"] = city["ville"].replace("_", " ").title()
    return city


def villes_disponibles(base: Path | str = SNAPSHOT_DIR) -> dict[str, str]:
    """Snapshots présents : {dossier: nom affiché}. Le dossier de base (Strasbourg) d'abord, puis base/<ville>/."""
    base = Path(base)
    dossiers = [base] + sorted(d for d in base.glob("*") if d.is_dir()) if base.is_dir() else [base]
    villes = {str(d): load_city(d)["titre"] for d in dossiers if (d / "gold_carreau_access.parquet").exists()}
    return villes or {str(base): load_city(base)["titre"]}


def load_snapshot(directory: Path | str = SNAPSHOT_DIR) -> dict[str, pd.DataFrame | None]:
    """Lit data/snapshot/<table>.parquet ; une table absente vaut None (l'application s'adapte)."""
    directory = Path(directory)
    out: dict[str, pd.DataFrame | None] = {}
    for name in TABLES:
        p = directory / f"{name}.parquet"
        out[name] = pd.read_parquet(p) if p.exists() else None
    return out


# --------------------------------------------------------------------------- objectif 1


def has_option(access: pd.DataFrame, modes=MODES) -> pd.Series:
    cols = [f"n_{m}" for m in modes if f"n_{m}" in access]
    if not cols:
        return pd.Series(False, index=access.index)
    return access[cols].fillna(0).sum(axis=1) > 0


def carreau_proche(access: pd.DataFrame | None, lat: float, lon: float, max_m: float = 150.0) -> pd.Series | None:
    """Carreau habité qui contient un point : centre le plus proche, à moins de `max_m` (demi-diagonale ≈ 141 m).
    None si le point est hors des carreaux habités du territoire."""
    if access is None or access.empty:
        return None
    dy = (access["lat"].to_numpy(float) - lat) * 111_320
    dx = (access["lon"].to_numpy(float) - lon) * 111_320 * math.cos(math.radians(lat))
    d = np.hypot(dx, dy)
    i = int(np.argmin(d))
    return access.iloc[i] if d[i] <= max_m else None


def options_carreau(row: pd.Series) -> str:
    """Options à 5 min d'un carreau en clair (« 2 arrêts de tram, 1 station Citiz »). Pas de population : seulement
    l'offre, qui n'est pas une donnée personnelle (carreaux protégés compris)."""
    parts = []
    for t, col in (("tram", "n_tram"), ("bus", "n_bus"), ("citiz", "n_citiz"), ("velhop", "n_velhop")):
        n = int(round(float(row.get(col, 0) or 0)))
        if n > 0:
            parts.append(f"{n} {TYPE_SINGULIERS[t] if n == 1 else TYPE_PLURIELS[t]}")
    return ", ".join(parts) if parts else "aucune option"


def access_kpis(access: pd.DataFrame) -> dict[str, float]:
    """Les 4 chiffres clés de l'objectif 1, pondérés par la population (tous carreaux, protégés compris)."""
    pop = access["pop"].fillna(0)
    total = float(pop.sum())
    any_opt = float(pop[has_option(access)].sum())
    citiz = float(pop[access["n_citiz"].fillna(0) > 0].sum())
    return {
        "population": total,
        "part_au_moins_une": any_opt / total if total else 0.0,
        "part_aucune": 1 - any_opt / total if total else 0.0,
        "part_citiz": citiz / total if total else 0.0,
    }


def classes_for_modes(access: pd.DataFrame, modes) -> pd.Series:
    """Classe d'accessibilité. Tous les modes : la classe officielle de gold_carreau_access.
    Sous-ensemble de modes : même règle (1 = aucune option ; 2 à 5 = quartiles des autres carreaux)
    recalculée sur le nombre d'options des modes choisis. Seuils = quartiles des valeurs : deux carreaux
    à égalité tombent toujours dans la même classe."""
    modes = tuple(m for m in MODES if m in modes)
    if set(modes) == set(MODES) and "classe" in access:
        return access["classe"].astype(int)
    n = access[[f"n_{m}" for m in modes]].fillna(0).sum(axis=1) if modes else pd.Series(0, index=access.index)
    classe = pd.Series(1, index=access.index, dtype=int)
    pos = n > 0
    if pos.sum() >= 4:
        seuils = n[pos].quantile([0.25, 0.5, 0.75]).to_numpy()
        classe[pos] = 2 + (n[pos].to_numpy()[:, None] > seuils).sum(axis=1)
    elif pos.any():
        classe[pos] = 3
    return classe


def walk_zones(plan: pd.DataFrame, candidates: pd.DataFrame, access: pd.DataFrame) -> list[dict]:
    """Zone réellement atteignable en 5 minutes à pied autour de chaque emplacement du plan : réunion des
    carreaux de 200 m de `covered_carreaux` (calculés sur le réseau piéton), pas un cercle de 400 m.
    Renvoie une liste de {rank, type, rings} ; rings = contours [[lat, lon], ...] à tracer en pointillés."""
    from shapely.geometry import box
    from shapely.ops import unary_union

    centre = access.set_index("carreau_id")[["lat", "lon"]]
    covered = candidates.set_index("candidate_id")["covered_carreaux"]
    dlat = 100.5 / 111_320  # 100,5 m : les carrés voisins se touchent et fusionnent
    zones = []
    for r in plan.itertuples():
        ids = [c for c in (covered.get(r.candidate_id) if r.candidate_id in covered.index else []) if c in centre.index]
        if not len(ids):
            continue
        pts = centre.loc[ids]
        dlon = dlat / np.cos(np.radians(pts["lat"].to_numpy()))
        shape = unary_union([box(lo - dl, la - dlat, lo + dl, la + dlat)
                             for la, lo, dl in zip(pts["lat"], pts["lon"], dlon)])
        polys = getattr(shape, "geoms", [shape])
        rings = [[[y, x] for x, y in poly.exterior.coords] for poly in polys]
        zones.append({"rank": int(r.rank), "type": r.type, "rings": rings})
    return zones


def map_carreaux(access: pd.DataFrame, modes=MODES) -> pd.DataFrame:
    """Données de la carte de l'objectif 1, avec carré de 200 m et info-bulle.
    Carreaux protégés : gris, info-bulle sans aucune valeur."""
    df = access[["carreau_id", "lat", "lon"]].copy()
    classe = classes_for_modes(access, modes)
    protected = access["i_est_200"].fillna(0).astype(int) == 1
    df["classe"] = classe
    df["protege"] = protected
    df["fill"] = [hex_to_rgb(PROTECTED_COLOR, 150) if p else hex_to_rgb(CLASS_COLORS[c], 175) for c, p in zip(classe, protected)]
    # Carré de 200 m centré sur le carreau (approximation locale, suffisante à l'échelle de l'affichage).
    dlat = 100 / 111_320
    dlon = 100 / (111_320 * np.cos(np.radians(df["lat"].to_numpy())))
    lat, lon = df["lat"].to_numpy(), df["lon"].to_numpy()
    df["polygon"] = [
        [[lo - dl, la - dlat], [lo + dl, la - dlat], [lo + dl, la + dlat], [lo - dl, la + dlat]]
        for la, lo, dl in zip(lat, lon, dlon)
    ]
    tips = []
    for i, (c, p) in enumerate(zip(classe, protected)):
        if p:
            tips.append(f"<b>{PROTECTED_LABEL}</b><br/>Aucune valeur affichée (moins de 11 ménages)")
            continue
        r = access.iloc[i]
        detail = " · ".join(f"{MODE_LABELS[m].split(' ')[0]} {int(r.get(f'n_{m}', 0) or 0)}" for m in MODES)
        tips.append(f"<b>{CLASS_LABELS[int(c)]}</b><br/>{int(round(r['pop'] or 0)):,} habitants<br/>{detail}".replace(",", " "))
    df["tooltip"] = tips
    return df


# --------------------------------------------------------------------------- objectif 2


def top_stations(stations: pd.DataFrame, types=None, n: int = 10) -> pd.DataFrame:
    df = stations if types is None else stations[stations["target_type"].isin(types)]
    return df.sort_values("pop_desservie", ascending=False).head(n)


# --------------------------------------------------------------------------- objectif 3


def _as_list(v) -> list:
    if v is None:
        return []
    if isinstance(v, float) and math.isnan(v):
        return []
    if isinstance(v, str):
        return [s.strip() for s in v.strip("[]").replace("'", "").replace('"', "").split(",") if s.strip()]
    return list(v)


def covered_by_type(access: pd.DataFrame) -> dict[str, set]:
    """Carreaux qui ont déjà une option du même type (règle de new_pop, DATA_CONTRACT : bus = ni bus ni tram)."""
    a = access.set_index("carreau_id")
    return {
        "citiz": set(a.index[a["n_citiz"].fillna(0) > 0]),
        "velhop": set(a.index[a["n_velhop"].fillna(0) > 0]),
        "bus": set(a.index[(a["n_bus"].fillna(0) + a["n_tram"].fillna(0)) > 0]),
    }


def gained_carreaux(plan: pd.DataFrame, candidates: pd.DataFrame, access: pd.DataFrame) -> pd.DataFrame:
    """Carreaux nouvellement desservis par le plan : (carreau_id, type) avec le rang du premier emplacement
    qui les dessert. Un carreau compte une fois par type (cohérent avec new_pop de l'optimiseur)."""
    if plan is None or plan.empty:
        return pd.DataFrame(columns=["carreau_id", "type", "rank"])
    already = covered_by_type(access)
    cov = candidates.set_index("candidate_id")["covered_carreaux"]
    rows, seen = [], {t: set(s) for t, s in already.items()}
    for r in plan.sort_values("rank").itertuples():
        t = r.type
        seen.setdefault(t, set())
        for c in _as_list(cov.get(r.candidate_id)):
            if c not in seen[t]:
                seen[t].add(c)
                rows.append((c, t, r.rank))
    return pd.DataFrame(rows, columns=["carreau_id", "type", "rank"])


def qpv_flag(qpv: pd.DataFrame | None) -> pd.Series | None:
    """Part de chaque carreau en QPV (0 à 1), à partir de gold_carreau_qpv quelle que soit la colonne retenue."""
    if qpv is None or qpv.empty or "carreau_id" not in qpv:
        return None
    for col in ("part_qpv", "share_qpv", "in_qpv", "qpv", "est_qpv", "is_qpv"):
        if col in qpv:
            s = pd.to_numeric(qpv[col], errors="coerce")
            if s.dropna().empty and qpv[col].dtype == object:
                continue
            return s.fillna(0).clip(0, 1).groupby(qpv["carreau_id"]).max()
    if "code_qpv" in qpv:
        return qpv.assign(f=qpv["code_qpv"].notna().astype(float)).groupby("carreau_id")["f"].max()
    return None


def niveau_vie_moyen(carreaux: pd.DataFrame, ids=None) -> float | None:
    """Niveau de vie moyen agrégé (somme des niveaux de vie / habitants) sur un groupe de carreaux.
    None si le groupe est trop petit pour être affiché (secret statistique)."""
    df = carreaux if ids is None else carreaux[carreaux["carreau_id"].isin(set(ids))]
    df = df.dropna(subset=["niveau_vie_somme", "pop"])
    if len(df) < MIN_CARREAUX_AGREGAT:
        return None
    if "menages" in df and df["menages"].sum() < MIN_MENAGES_AGREGAT:
        return None
    pop = df["pop"].sum()
    return float(df["niveau_vie_somme"].sum() / pop) if pop > 0 else None


def qpv_reperes(access: pd.DataFrame, qpv: pd.DataFrame | None) -> dict[str, float | None]:
    """Repères pour lire la part en QPV d'un plan : part des habitants vivant en QPV dans toute la population,
    et parmi ceux qui ont aujourd'hui une station Citiz à 5 min. Agrégés seulement (secret statistique)."""
    out: dict[str, float | None] = {"population": None, "citiz": None}
    flag = qpv_flag(qpv)
    if flag is None:
        return out
    a = pd.DataFrame({"pop": access["pop"].fillna(0).to_numpy(),
                      "f": flag.reindex(access["carreau_id"]).fillna(0).to_numpy(),
                      "citiz": (access["n_citiz"].fillna(0) > 0).to_numpy()})
    for key, grp in (("population", a), ("citiz", a[a["citiz"]])):
        if len(grp) >= MIN_CARREAUX_AGREGAT and grp["pop"].sum() > 0:
            out[key] = float((grp["pop"] * grp["f"]).sum() / grp["pop"].sum())
    return out


def qpv_par_type(plan: pd.DataFrame, candidates: pd.DataFrame, access: pd.DataFrame,
                 qpv: pd.DataFrame | None) -> dict[str, float | None]:
    """Part en QPV des habitants nouvellement desservis, type d'équipement par type.
    None pour un type qui dessert trop peu de carreaux pour être affiché (secret statistique)."""
    flag = qpv_flag(qpv)
    g = gained_carreaux(plan, candidates, access)
    if flag is None or g.empty:
        return {}
    pop = access.set_index("carreau_id")["pop"].fillna(0)
    out: dict[str, float | None] = {}
    for t in [t for t in ("citiz", "bus", "velhop") if t in set(g["type"])]:
        ids = g.loc[g["type"] == t, "carreau_id"].unique()
        p = pop.reindex(ids).fillna(0)
        ok = len(ids) >= MIN_CARREAUX_AGREGAT and p.sum() > 0
        out[t] = float((p * flag.reindex(ids).fillna(0)).sum() / p.sum()) if ok else None
    return out


def couts_unitaires(costs: pd.DataFrame | None, horizon: int, types=("citiz", "bus", "velhop")) -> dict[str, float]:
    """Coût d'un emplacement sur l'horizon (investissement + horizon × fonctionnement annuel), règle de l'optimiseur."""
    if costs is None or costs.empty:
        return {}
    c = costs.set_index("type")
    return {t: float(c.loc[t, "capex_eur"] + horizon * c.loc[t, "opex_annuel_eur"]) for t in types if t in c.index}


def co2_tonnes(n_citiz: int) -> int:
    """CO₂ évité, arrondi à l'unité (t CO₂e / an), voitures Citiz seulement."""
    return int(round(n_citiz * CO2_T_PER_CITIZ_CAR))


def plan_indicators(
    plan: pd.DataFrame,
    candidates: pd.DataFrame,
    access: pd.DataFrame,
    carreaux: pd.DataFrame | None = None,
    qpv: pd.DataFrame | None = None,
) -> dict:
    """Synthèse chiffrée d'un plan. Les chiffres sociaux sont agrégés sur l'ensemble des carreaux gagnés.
    Chiffre principal : `habitants_uniques`, les personnes distinctes nouvellement desservies (dénominateur de
    toutes les parts). `habitants_gagnes` = somme des new_pop de l'optimiseur : un habitant qui gagne à la fois
    le bus et Citiz y compte deux fois, ce n'est jamais le chiffre affiché."""
    out: dict = {
        "n_emplacements": 0, "cout_eur": 0.0, "habitants_gagnes": 0.0, "n_par_type": {},
        "voitures_citiz": 0, "co2_t_an": 0, "habitants_uniques": 0.0, "cout_par_habitant": None, "part_classes_1_2": None,
        "part_qpv": None, "niveau_vie_gagnes": None, "niveau_vie_population": None, "niveau_vie_citiz": None,
    }
    if carreaux is not None:
        out["niveau_vie_population"] = niveau_vie_moyen(carreaux)
        out["niveau_vie_citiz"] = niveau_vie_moyen(carreaux, access.loc[access["n_citiz"].fillna(0) > 0, "carreau_id"])
    if plan is None or plan.empty:
        return out
    counts = plan["type"].value_counts().to_dict()
    out.update(
        n_emplacements=len(plan),
        cout_eur=float(plan["cost_eur"].sum()),
        habitants_gagnes=float(plan["new_pop"].sum()),
        n_par_type={t: int(counts.get(t, 0)) for t in ("citiz", "bus", "velhop")},
        voitures_citiz=int(counts.get("citiz", 0)),  # hypothèse prudente : 1 voiture par nouvelle station
        co2_t_an=co2_tonnes(int(counts.get("citiz", 0))),
    )
    g = gained_carreaux(plan, candidates, access)
    ids = g["carreau_id"].unique()
    pop = access.set_index("carreau_id")["pop"].fillna(0)
    pop_g = pop.reindex(ids).fillna(0)
    total = float(pop_g.sum())
    out["habitants_uniques"] = total
    if total > 0:
        out["cout_par_habitant"] = out["cout_eur"] / total
        cl = access.set_index("carreau_id")["classe"].reindex(ids)
        out["part_classes_1_2"] = float(pop_g[cl.isin([1, 2]).to_numpy()].sum() / total)
        flag = qpv_flag(qpv)
        if flag is not None and len(ids) >= MIN_CARREAUX_AGREGAT:
            out["part_qpv"] = float((pop_g * flag.reindex(ids).fillna(0)).sum() / total)
    if carreaux is not None:
        out["niveau_vie_gagnes"] = niveau_vie_moyen(carreaux, ids)
    return out


def plan_table(plan: pd.DataFrame, predictions: pd.DataFrame | None = None) -> pd.DataFrame:
    """Tableau priorisé affiché et exporté en CSV."""
    df = plan.sort_values("rank").copy()
    if predictions is not None and "p_usage" in predictions:
        df = df.merge(predictions[["candidate_id", "p_usage"]], on="candidate_id", how="left")
    cols = {
        "rank": "Rang", "type": "Équipement", "candidate_id": "Emplacement", "lat": "Latitude", "lon": "Longitude",
        "new_pop": "Habitants nouvellement desservis", "p_usage": P_USAGE_LABEL,
        "cost_eur": "Coût (€)", "cumul_pop": "Habitants cumulés", "cumul_cost_eur": "Coût cumulé (€)",
    }
    df["type"] = df["type"].map(TYPE_LABELS).fillna(df["type"])
    keep = [c for c in cols if c in df]
    return df[keep].rename(columns=cols)


def fmt_int(x: float) -> str:
    """12 345 (espace insécable fine à la française)."""
    return f"{int(round(x)):,}".replace(",", " ")


def plan_table_fr(table: pd.DataFrame) -> pd.DataFrame:
    """Version affichée du tableau du plan : nombres au format français (le CSV garde les valeurs numériques)."""
    df = table.copy()
    for col in ("Coût (€)", "Coût cumulé (€)"):
        if col in df:
            df[col] = df[col].map(fmt_eur_plein)
    for col in ("Habitants nouvellement desservis", "Habitants cumulés"):
        if col in df:
            df[col] = df[col].map(fmt_int)
    if P_USAGE_LABEL in df:
        df[P_USAGE_LABEL] = df[P_USAGE_LABEL].map(lambda v: "—" if pd.isna(v) else fmt_dec(v, 2))
    for col in ("Latitude", "Longitude"):
        if col in df:
            df[col] = df[col].map(lambda v: fmt_dec(v, 5))
    return df


def compte_types(n_par_type: dict[str, int]) -> str:
    """« 15 arrêts de bus, 3 stations Citiz » (types absents omis, singulier pour 1)."""
    return ", ".join(f"{n} {TYPE_PLURIELS[t] if n > 1 else TYPE_SINGULIERS[t]}"
                     for t in ("bus", "citiz", "velhop") if (n := int(n_par_type.get(t, 0))))


def recommandation(kp: dict, budget: float, horizon: int) -> str:
    """Phrase de synthèse du plan, en tête de l'onglet ③."""
    n = kp["n_emplacements"]
    phrase = (f"Avec {fmt_eur_plein(budget)} sur {horizon} an{'s' if horizon > 1 else ''}, ouvrir "
              f"{n} emplacement{'s' if n > 1 else ''} ({compte_types(kp['n_par_type'])}) dessert "
              f"{fmt_int(kp['habitants_uniques'])} habitants qui n'avaient pas cette option à 5 minutes à pied")
    if kp.get("cout_par_habitant"):
        phrase += f", soit {fmt_int(kp['cout_par_habitant'])}\u00a0€ par habitant desservi"
    return phrase + "."


def conclusion_equite(ke: dict, kq: dict) -> str:
    """Conclusion de la comparaison (ke : chaque habitant compte pareil, kq : équité), une décimale partout."""
    # écart entre les chiffres arrondis affichés, pour que la soustraction tombe juste à l'écran
    base = round(ke["habitants_uniques"])
    d = round(kq["habitants_uniques"]) - base
    rel = f" ({'−' if d < 0 else '+'}{fmt_pct(abs(d) / base)})" if base else ""
    if d < 0:
        phrase = f"L'équité coûte {fmt_int(-d)} habitants nouvellement desservis{rel}"
    elif d > 0:
        phrase = f"L'équité dessert {fmt_int(d)} habitants de plus{rel}"
    else:
        phrase = "L'équité ne change pas le nombre d'habitants nouvellement desservis"
    effets = []
    for cle, nom in (("part_qpv", "la part en quartier prioritaire (QPV)"),
                     ("part_classes_1_2", "la part dans les zones les moins desservies (classes 1 et 2)")):
        a, b = ke.get(cle), kq.get(cle)
        if a is not None and b is not None:
            effets.append((a, b, nom))
    if not effets:
        return phrase + "."
    morceaux, precedent = [], None
    for a, b, nom in effets:
        verbe = "augmente" if b > a else ("réduit" if b < a else "ne change pas")
        valeurs = f"de {fmt_pct(a)} à {fmt_pct(b)}" if b != a else f"({fmt_pct(a)})"
        morceaux.append((f"{verbe} " if verbe != precedent else "") + f"{nom} {valeurs}")
        precedent = verbe
    return f"{phrase} et {', '.join(morceaux)}."


def plan_diff(plan_a: pd.DataFrame, plan_b: pd.DataFrame) -> tuple[set, set]:
    """Emplacements présents dans un seul des deux plans : (seulement dans a, seulement dans b)."""
    a = set(plan_a["candidate_id"]) if "candidate_id" in plan_a else set()
    b = set(plan_b["candidate_id"]) if "candidate_id" in plan_b else set()
    return a - b, b - a


def message_plan_vide(couts: dict[str, float], budget: float, horizon: int, budgets=()) -> str:
    """Plan vide : coût d'un emplacement sur l'horizon pour chaque type choisi et budget minimum à prévoir."""
    if not couts:
        return "Aucun emplacement ne tient dans ce budget."
    detail = " ; ".join(f"{TYPE_LABELS[t]} : {fmt_eur_plein(c)}" for t, c in sorted(couts.items(), key=lambda x: x[1]))
    mini = min(couts.values())
    msg = f"Aucun emplacement ne tient dans ce budget. Coût d'un emplacement sur {horizon} an{'s' if horizon > 1 else ''} : {detail}. "
    if budget >= mini:
        return msg + ("Le budget suffit pour au moins un emplacement : aucun emplacement candidat ne passe les "
                      "filtres choisis (ressemblance minimale aux stations Citiz actuelles, types d'équipement).")
    proposes = [b for b in budgets if b >= mini]
    return msg + f"Budget minimum : {fmt_eur_plein(mini)}" + (f" (choisir au moins {fmt_eur(proposes[0])})." if proposes else ".")


def fmt_eur_plein(x: float) -> str:
    """500 000 € (montant entier, sans abréviation)."""
    return f"{fmt_int(x)}\u00a0€"


def fmt_dec(x: float, digits: int = 1) -> str:
    """0,5 (virgule décimale)."""
    return f"{x:.{digits}f}".replace(".", ",")


def fmt_eur(x: float) -> str:
    if x >= 1_000_000:
        return f"{x / 1_000_000:.1f}\u00a0M€".replace(".", ",")
    return f"{x / 1000:.0f}\u00a0k€"


def fmt_pct(x: float | None, digits: int = 1) -> str:
    return "—" if x is None else f"{100 * x:.{digits}f}\u00a0%".replace(".", ",")
