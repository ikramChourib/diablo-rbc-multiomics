"""
================================================================================
 diablo_core -- logique réutilisable pour l'application Streamlit DIABLO + RBC
================================================================================

Ce module est un refactor, sans effet de bord, de `diablo_compare_v3.py`
(pipeline de comparaison DIABLO omique seule vs. omique + histopathologie,
avec un classifieur RBC -- raisonnement à base de cas -- opérant dans
l'espace latent de DIABLO). Toute la logique mathématique / statistique
(BlockPLSDA dense et sparse, CaseBase, normalisation de coloration Reinhard,
métriques) est conservée à l'identique ; seules les entrées/sorties changent
(DataFrames et objets fichier en mémoire plutôt que chemins codés en dur, et
un seed choisi à l'appel plutôt qu'un `np.random.seed` global à l'import).

RBC dans l'espace latent DIABLO -- rappel méthodologique important (à citer
dans l'article) :
  Le RBC ne retrouve pas ses cas voisins dans les variables omiques brutes,
  mais dans les SCORES LATENTS à 2 dimensions par bloc appris par
  block.plsda (dense) -- exactement l'espace que DIABLO utilise lui-même
  pour classer via ses centroïdes. C'est un k-NN pondéré par bloc, appliqué
  à une représentation déjà optimisée pour séparer les sous-types, au lieu
  des données brutes.

  ATTENTION -- implication méthodologique : le RBC n'est alors plus un
  algorithme totalement indépendant de DIABLO, c'est un raisonnement à base
  de cas appliqué à la représentation apprise par DIABLO (une forme
  d'"apprentissage de métrique" supervisé avant k-NN, technique courante et
  légitime). Le RBC garde son cycle Retrieve/Reuse et reste interprétable
  (on peut toujours citer les patientes les plus proches), mais la
  comparaison "RBC vs DIABLO" doit être présentée comme une comparaison
  entre deux stratégies de DÉCISION sur la même représentation apprise
  (centroïde de classe vs. vote pondéré des k plus proches voisins), pas
  comme deux méthodes indépendantes "from scratch".
================================================================================
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import ndimage
from sklearn.model_selection import StratifiedShuffleSplit

try:
    import openslide
    OPENSLIDE_AVAILABLE = True
except ImportError:
    OPENSLIDE_AVAILABLE = False


ProgressCB = Callable[[float, str], None]


def _tick(cb: ProgressCB | None, frac: float, msg: str) -> None:
    if cb is not None:
        cb(min(max(frac, 0.0), 1.0), msg)


COLOR_OMICS = "#1F77B4"
COLOR_OMICS_IMAGE = "#9467BD"

IMAGE_FEATURE_NAMES = [
    "frac_tissue", "R_mean", "G_mean", "B_mean", "R_std", "G_std", "B_std",
    "gray_mean", "gray_std", "texture_std", "sat_mean", "sat_std",
]

REPEATED_N_REPEATS = 20
REPEATED_TEST_SIZE = 70
REPEATED_SEED = 0
REPEATED_TOTAL_PATIENTS = 220

MODEL_TYPE_LABELS = {"dense": "Modèle dense", "sparse": "Modèle sparse", "rbc": "RBC (Retrieve/Reuse)"}
CONFIG_ORDER = ["A) omique seule", "B) omique+image"]
CONFIG_SHORT = {"A) omique seule": "Omique seule", "B) omique+image": "Omique + image"}


# ------------------------------------------------------------------------
# 1) DONNÉES OMIQUES -- synthétiques (démo) ou fournies par l'utilisateur
# ------------------------------------------------------------------------

def _sniff_delimiter(path_or_buffer) -> str:
    """Détecte le séparateur d'un CSV (virgule, point-virgule, tabulation,
    barre verticale) à partir de sa première ligne, pour tolérer les
    exports 'européens' (Excel utilise souvent ';') sans faire planter la
    lecture ni, pire, produire silencieusement un bloc à 0 colonne (toute
    la ligne lue comme un seul champ, consommé par l'index)."""
    candidates = [",", ";", "\t", "|"]
    if hasattr(path_or_buffer, "read"):
        pos = path_or_buffer.tell() if hasattr(path_or_buffer, "tell") else 0
        raw = path_or_buffer.read(8192)
        if hasattr(path_or_buffer, "seek"):
            path_or_buffer.seek(pos)
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="replace")
        sample = raw
    else:
        with open(path_or_buffer, "r", encoding="utf-8", errors="replace") as f:
            sample = f.read(8192)
    lines = sample.splitlines()
    first_line = lines[0] if lines else ""
    counts = {c: first_line.count(c) for c in candidates}
    best = max(counts, key=counts.get)
    return best if counts[best] > 0 else ","


def _smart_read_csv(path_or_buffer, index_col=0, label: str = "fichier") -> pd.DataFrame:
    """`pd.read_csv` avec détection automatique du séparateur, et une erreur
    claire (plutôt qu'un plantage cryptique plus tard dans l'ajustement du
    modèle) si le fichier ne produit aucune colonne de variables -- le
    symptôme typique d'un séparateur mal détecté ou d'un format inattendu."""
    sep = _sniff_delimiter(path_or_buffer)
    if hasattr(path_or_buffer, "seek"):
        path_or_buffer.seek(0)
    df = pd.read_csv(path_or_buffer, index_col=index_col, sep=sep, engine="python")
    if df.shape[1] == 0:
        raise ValueError(
            f"Le fichier « {label} » ne contient AUCUNE colonne de variables "
            f"après lecture (séparateur détecté : {sep!r}). Vérifie qu'il "
            "s'agit bien d'un CSV avec les identifiants patientes en première "
            "colonne puis les variables dans les colonnes suivantes, et que "
            "le séparateur (virgule, point-virgule, tabulation) est cohérent "
            "sur tout le fichier."
        )
    return df

def generate_synthetic_breast_tcga(
    n_train: int = 150, n_test: int = 70, seed: int = 42,
) -> tuple[dict[str, pd.DataFrame], pd.Series, dict[str, pd.DataFrame], pd.Series]:
    """Génère un jeu synthétique qui imite le cas d'usage TCGA-BRCA de
    mixOmics : 3 blocs omiques (mrna/mirna/protein) en train, 2 seulement
    (mrna/mirna) en test -- le bloc protéines n'existe que dans le
    tutoriel officiel côté train, jamais côté test."""
    rng = np.random.default_rng(seed)
    classes = ["Basal", "Her2", "LumA"]
    block_shapes = {"mrna": 200, "mirna": 184, "protein": 142}

    n_per_class_train = n_train // len(classes)
    labels_train = np.array([c for c in classes for _ in range(n_per_class_train)])
    # complète pour retomber exactement sur n_train si non multiple de 3
    while len(labels_train) < n_train:
        labels_train = np.append(labels_train, classes[len(labels_train) % len(classes)])

    n_per_class_test = n_test // len(classes)
    reste = n_test - n_per_class_test * len(classes)
    counts_test = [n_per_class_test + (1 if i < reste else 0) for i in range(len(classes))]
    labels_test = np.array(
        [c for c, n in zip(classes, counts_test) for _ in range(n)], dtype=object
    )

    train_ids = [f"S{i:03d}" for i in range(n_train)]
    test_ids = [f"T{i:03d}" for i in range(n_test)]

    train = {}
    for block_name, p in block_shapes.items():
        block = np.zeros((n_train, p))
        for i, label in enumerate(labels_train):
            class_mean = {"Basal": 1.0, "Her2": 0.0, "LumA": -1.0}[label]
            block[i] = rng.normal(class_mean, 1.0, size=p)
        train[block_name] = pd.DataFrame(block, index=train_ids,
                                          columns=[f"{block_name}_{j}" for j in range(p)])

    test = {}
    for block_name, p in {"mrna": 200, "mirna": 184}.items():
        block = np.zeros((n_test, p))
        for i, label in enumerate(labels_test):
            class_mean = {"Basal": 1.1, "Her2": 0.1, "LumA": -1.1}[label]
            block[i] = rng.normal(class_mean, 1.0, size=p)
        test[block_name] = pd.DataFrame(block, index=test_ids,
                                         columns=[f"{block_name}_{j}" for j in range(p)])

    y_train = pd.Series(labels_train, index=train_ids, name="subtype")
    y_test = pd.Series(labels_test, index=test_ids, name="subtype")
    return train, y_train, test, y_test


def _read_subtype(
    path_or_buffer, label: str = "fichier subtype",
    fallback_ids: list[str] | None = None, messages: list[str] | None = None,
) -> pd.Series:
    """Lit un fichier de sous-types. Deux formats acceptés :
      1) identifiant patiente en première colonne (index) + une colonne de
         sous-type -- alignement fiable PAR ID dans `_align_blocks_and_labels`.
      2) une seule colonne de sous-types, SANS colonne d'identifiant (p.ex.
         un export brut de type PAM50/sous-type dans l'ordre du fichier
         mRNA). Dans ce cas, faute d'ID, les étiquettes sont associées PAR
         POSITION à `fallback_ids` (l'index du bloc mRNA correspondant) --
         mais, contrairement à l'ancien bug, ceci est toujours signalé
         EXPLICITEMENT (jamais silencieux) dans `messages`, et seulement
         utilisé quand aucune colonne d'ID n'est réellement présente."""
    msgs = messages if messages is not None else []
    sep = _sniff_delimiter(path_or_buffer)
    if hasattr(path_or_buffer, "seek"):
        path_or_buffer.seek(0)
    df = pd.read_csv(path_or_buffer, index_col=0, sep=sep, engine="python")
    if df.shape[1] >= 1:
        s = df.iloc[:, 0]
        s.index = s.index.astype(str)
        return s

    # 0 colonne restante après avoir pris la 1re colonne comme index : le
    # fichier n'a en réalité qu'UNE SEULE colonne au total (pas d'ID).
    if hasattr(path_or_buffer, "seek"):
        path_or_buffer.seek(0)
    df2 = pd.read_csv(path_or_buffer, sep=sep, engine="python")
    if df2.shape[1] == 1:
        values = df2.iloc[:, 0]
        col_name = df2.columns[0]
    elif "subtype" in df2.columns:
        values = df2["subtype"]
        col_name = "subtype"
    else:
        raise ValueError(
            f"Le fichier « {label} » ne contient aucune colonne de sous-type "
            f"exploitable (séparateur détecté : {sep!r}). Attendu : soit un "
            "identifiant patiente en première colonne puis une colonne de "
            "sous-type, soit une seule colonne de sous-types (dans le même "
            "ordre que le fichier mRNA correspondant)."
        )
    if fallback_ids is not None and len(values) == len(fallback_ids):
        msgs.append(
            f"[alignement][ATTENTION] « {label} » (colonne « {col_name} ») ne "
            "contient AUCUN identifiant patiente -- les sous-types ont été "
            f"associés PAR POSITION à l'ordre des {len(fallback_ids)} lignes "
            "du fichier mRNA correspondant. Vérifiez que les deux fichiers "
            "sont bien dans le même ordre ; pour un alignement fiable, "
            "ajoutez une colonne d'identifiant patiente à ce fichier."
        )
        s = pd.Series(values.values, index=list(fallback_ids), name=col_name)
    elif fallback_ids is not None:
        raise ValueError(
            f"Le fichier « {label} » n'a pas de colonne d'identifiant "
            f"patiente et contient {len(values)} ligne(s), alors que le "
            f"fichier mRNA correspondant en a {len(fallback_ids)} -- "
            "impossible de les associer par position. Ajoutez une colonne "
            "d'identifiant patiente à ce fichier, ou corrigez son nombre de "
            "lignes."
        )
    else:
        s = values
    s.index = s.index.astype(str)
    return s


def _align_blocks_and_labels(
    blocks: dict[str, pd.DataFrame], y: pd.Series, split: str, messages: list[str],
) -> tuple[dict[str, pd.DataFrame], pd.Series, list[str]]:
    """Aligne STRICTEMENT par identifiant patiente (l'index de chaque CSV,
    jamais sa position de ligne) tous les blocs fournis et le vecteur
    d'étiquettes correspondant.

    Correction d'un bug important : la version précédente écrasait
    l'index du fichier `subtype` par celui du bloc mRNA dès que les deux
    fichiers avaient le même NOMBRE de lignes, sans vérifier qu'ils
    étaient dans le même ORDRE. Si le fichier subtype d'une utilisatrice
    réelle liste les patientes dans un ordre différent (p.ex. trié
    alphabétiquement) de celui du fichier mRNA, chaque étiquette se
    retrouvait alors silencieusement associée à la mauvaise patiente --
    sans aucune erreur, juste un entraînement/une évaluation sur un
    alignement erroné. Ici, on ne fait plus JAMAIS cette hypothèse : on
    prend l'intersection des identifiants réellement présents dans tous
    les fichiers, on réindexe tout sur cette liste commune (triée), et on
    signale clairement tout identifiant écarté."""
    aligned_blocks: dict[str, pd.DataFrame] = {}
    for name, df in blocks.items():
        df = df.copy()
        df.index = df.index.astype(str)
        aligned_blocks[name] = df

    y = y.copy()
    y.index = y.index.astype(str)

    common = set(y.index)
    for df in aligned_blocks.values():
        common &= set(df.index)
    common_sorted = sorted(common)

    if not common_sorted:
        exemples = {name: list(df.index[:3]) for name, df in aligned_blocks.items()}
        exemples["subtype"] = list(y.index[:3])
        raise ValueError(
            f"[alignement:{split}] Aucun identifiant patiente commun entre "
            f"les fichiers fournis -- impossible d'aligner les blocs et les "
            f"étiquettes. Premiers identifiants vus par fichier : {exemples}. "
            "Vérifie que tous les fichiers utilisent bien le même format "
            "d'identifiant (p.ex. 'TCGA-XX-XXXX' partout, ou le même "
            "suffixe court partout), et le bon séparateur de colonnes."
        )

    dropped_report = []
    for name, df in aligned_blocks.items():
        extra = sorted(set(df.index) - common)
        if extra:
            dropped_report.append((name, extra))
    extra_y = sorted(set(y.index) - common)
    if extra_y:
        dropped_report.append(("subtype", extra_y))

    if dropped_report:
        total_dropped = sum(len(v) for _, v in dropped_report)
        detail = "; ".join(f"{name}: {len(ids)} id(s), ex. {ids[:3]}" for name, ids in dropped_report)
        messages.append(
            f"[alignement:{split}][ATTENTION] {total_dropped} identifiant(s) patiente "
            f"non commun(s) à TOUS les fichiers ({', '.join(aligned_blocks.keys())}, subtype) "
            f"ont été écartés par alignement strict par ID ({detail}). "
            f"{len(common_sorted)} patiente(s) conservée(s)."
        )
    else:
        messages.append(
            f"[alignement:{split}] Identifiants cohérents entre tous les fichiers "
            f"({len(common_sorted)} patiente(s), alignement strict par ID vérifié)."
        )

    aligned_blocks = {name: df.loc[common_sorted] for name, df in aligned_blocks.items()}
    aligned_y = y.loc[common_sorted]
    return aligned_blocks, aligned_y, common_sorted


def load_breast_tcga_from_uploads(
    train_mrna, train_mirna, train_subtype,
    test_mrna, test_mirna, test_subtype,
    train_protein=None, test_protein=None,
    messages: list[str] | None = None,
) -> tuple[dict[str, pd.DataFrame], pd.Series, dict[str, pd.DataFrame], pd.Series]:
    """Charge le jeu de données omiques à partir de fichiers (chemins ou
    objets file-like, p.ex. issus de `st.file_uploader`). Format attendu :
    chaque CSV a les identifiants patientes en première colonne (index),
    puis les variables en colonnes (blocs), ou une seule colonne `subtype`.
    Le bloc protéines (`train_protein`/`test_protein`) est OPTIONNEL, aussi
    bien côté train que côté test : s'il est omis, le pipeline s'exécute
    avec seulement mrna+mirna (et, le cas échéant, +image) pour le split
    concerné. Les blocs et les étiquettes sont alignés strictement par
    identifiant patiente (voir `_align_blocks_and_labels`) ; tout
    identifiant absent d'un des fichiers est écarté et signalé dans
    `messages`, jamais silencieusement mal aligné. Un fichier de sous-types
    sans colonne d'identifiant (une seule colonne de labels) est toléré --
    associé par position à l'ordre du fichier mRNA correspondant -- mais
    toujours signalé explicitement dans `messages` (voir `_read_subtype`)."""
    msgs = messages if messages is not None else []

    train_raw = {
        "mrna": _smart_read_csv(train_mrna, label="data.train_mrna.csv"),
        "mirna": _smart_read_csv(train_mirna, label="data.train_mirna.csv"),
    }
    if train_protein is not None:
        train_raw["protein"] = _smart_read_csv(train_protein, label="data.train_protein.csv")
    y_train_raw = _read_subtype(
        train_subtype, label="data.train_subtype.csv",
        fallback_ids=list(train_raw["mrna"].index), messages=msgs,
    )
    train, y_train, _ = _align_blocks_and_labels(train_raw, y_train_raw, "train", msgs)

    test_raw = {
        "mrna": _smart_read_csv(test_mrna, label="data.test_mrna.csv"),
        "mirna": _smart_read_csv(test_mirna, label="data.test_mirna.csv"),
    }
    if test_protein is not None:
        test_raw["protein"] = _smart_read_csv(test_protein, label="data.test_protein.csv")
    y_test_raw = _read_subtype(
        test_subtype, label="data.test_subtype.csv",
        fallback_ids=list(test_raw["mrna"].index), messages=msgs,
    )
    test, y_test, _ = _align_blocks_and_labels(test_raw, y_test_raw, "test", msgs)

    return train, y_train, test, y_test


# ------------------------------------------------------------------------
# 2) BLOC IMAGE -- normalisation de coloration + extraction de features
# ------------------------------------------------------------------------

def _detect_tissue_mask(thumb_rgb: np.ndarray) -> np.ndarray:
    arr = thumb_rgb.astype(np.float32)
    ecart = arr.max(axis=2) - arr.min(axis=2)
    lum = arr.mean(axis=2)
    mask = (lum < 235) & (ecart > 8)
    if mask.sum() == 0:
        return np.ones(thumb_rgb.shape[:2], dtype=bool)

    mask_dil = ndimage.binary_dilation(mask, iterations=2)
    labels, n = ndimage.label(mask_dil)
    if n == 0:
        return mask
    tailles = ndimage.sum(mask, labels, range(1, n + 1))
    plus_gros = int(np.argmax(tailles)) + 1
    return (labels == plus_gros) & mask


# ---- conversions colorimétriques sRGB <-> CIE LAB (from scratch, numpy) ---

_RGB2XYZ = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
])
_XYZ2RGB = np.array([
    [3.2404542, -1.5371385, -0.4985314],
    [-0.9692660, 1.8760108, 0.0415560],
    [0.0556434, -0.2040259, 1.0572252],
])
_WHITE = np.array([0.95047, 1.0, 1.08883])
_DELTA = 6.0 / 29.0


def _rgb_to_lab(rgb_uint8: np.ndarray) -> np.ndarray:
    rgb = rgb_uint8.astype(np.float64) / 255.0
    mask = rgb > 0.04045
    linear = np.where(mask, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)
    xyz = linear @ _RGB2XYZ.T
    xyz_n = xyz / _WHITE

    def f(t):
        return np.where(t > _DELTA ** 3, np.cbrt(t), t / (3 * _DELTA ** 2) + 4.0 / 29.0)

    fx, fy, fz = f(xyz_n[..., 0]), f(xyz_n[..., 1]), f(xyz_n[..., 2])
    L = 116 * fy - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)
    return np.stack([L, a, b], axis=-1)


def _lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    fy = (L + 16) / 116
    fx = fy + a / 500
    fz = fy - b / 200

    def finv(t):
        return np.where(t > _DELTA, t ** 3, 3 * _DELTA ** 2 * (t - 4.0 / 29.0))

    xyz = np.stack([_WHITE[0] * finv(fx), _WHITE[1] * finv(fy), _WHITE[2] * finv(fz)], axis=-1)
    linear = xyz @ _XYZ2RGB.T
    mask = linear > 0.0031308
    srgb = np.where(mask, 1.055 * np.power(np.clip(linear, 0, None), 1 / 2.4) - 0.055, 12.92 * linear)
    srgb = np.clip(srgb, 0, 1)
    return np.round(srgb * 255).astype(np.uint8)


def _lab_stats_tissue(thumb_rgb: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lab = _rgb_to_lab(thumb_rgb)
    pixels = lab[mask] if mask.sum() > 0 else lab.reshape(-1, 3)
    mean = pixels.mean(axis=0)
    std = pixels.std(axis=0)
    std[std < 1e-6] = 1e-6
    return mean, std


def _apply_reinhard(thumb_rgb: np.ndarray, src_mean: np.ndarray, src_std: np.ndarray,
                     tgt_mean: np.ndarray, tgt_std: np.ndarray) -> np.ndarray:
    """Recale les statistiques colorimétriques (moyenne/écart-type par canal
    L*a*b*) de l'image source vers celles de la référence -- normalisation de
    coloration de Reinhard, corrige le biais de centre/scanner H&E."""
    lab = _rgb_to_lab(thumb_rgb)
    lab_norm = (lab - src_mean) / src_std * tgt_std + tgt_mean
    return _lab_to_rgb(lab_norm)


def _extract_features_from_thumb(thumb_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Extrait les 12 features à partir d'une vignette RGB déjà (ou non)
    normalisée en coloration, avec un masque de tissu déjà calculé."""
    pixels = thumb_rgb[mask] if mask.sum() > 0 else thumb_rgb.reshape(-1, 3)

    gray = thumb_rgb.astype(np.float32).mean(axis=2)
    gray_tissue = gray[mask] if mask.sum() > 0 else gray.ravel()
    gy, gx = np.gradient(gray)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)
    texture_std = float(grad_mag[mask].std()) if mask.sum() > 0 else float(grad_mag.std())

    mx = thumb_rgb.max(axis=2).astype(np.float32)
    mn = thumb_rgb.min(axis=2).astype(np.float32)
    sat = np.divide(mx - mn, mx, out=np.zeros_like(mx), where=mx > 0)
    sat_tissue = sat[mask] if mask.sum() > 0 else sat.ravel()

    return np.array([
        float(mask.mean()),
        float(pixels[:, 0].mean()), float(pixels[:, 1].mean()), float(pixels[:, 2].mean()),
        float(pixels[:, 0].std()), float(pixels[:, 1].std()), float(pixels[:, 2].std()),
        float(gray_tissue.mean()), float(gray_tissue.std()),
        texture_std,
        float(sat_tissue.mean()), float(sat_tissue.std()),
    ])


def _load_thumb_and_mask(svs_path: Path, thumb_size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    slide = openslide.OpenSlide(str(svs_path))
    try:
        thumb = np.asarray(slide.get_thumbnail(thumb_size).convert("RGB"))
    finally:
        slide.close()
    mask = _detect_tissue_mask(thumb)
    return thumb, mask


def _index_svs_files(svs_dir: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    if svs_dir is None or not Path(svs_dir).exists():
        return index
    svs_dir = Path(svs_dir)
    for path in list(svs_dir.glob("*.svs")) + list(svs_dir.glob("*.SVS")):
        sample_id = path.stem.split("_", 1)[0]
        index[sample_id] = path
    return index


def _index_svs_uploads(uploads) -> dict[str, Any]:
    """Indexe une liste de fichiers uploadés (objets file-like avec un
    attribut `.name`, comme ceux de `st.file_uploader`) par identifiant
    patiente déduit du nom de fichier, comme `_index_svs_files` le fait
    pour un dossier local."""
    index: dict[str, Any] = {}
    if not uploads:
        return index
    for f in uploads:
        name = getattr(f, "name", None)
        if not name:
            continue
        stem = Path(name).stem
        sample_id = stem.split("_", 1)[0]
        index[sample_id] = f
    return index


def generate_synthetic_image_features(ids: list[str], y: pd.Series, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    class_shift = {"Basal": -1.0, "Her2": 0.0, "LumA": 1.0}
    rows = []
    for sid in ids:
        label = y.loc[sid] if sid in y.index else rng.choice(list(class_shift))
        shift = class_shift.get(label, 0.0)
        rows.append(rng.normal(shift, 1.0, size=len(IMAGE_FEATURE_NAMES)))
    return pd.DataFrame(rows, index=ids, columns=IMAGE_FEATURE_NAMES)


def _check_duplicate_slides(train_index: dict, test_index: dict, messages: list[str]) -> None:
    """Détecte des .svs identiques liés à plusieurs patientes (souvent un
    reliquat de liens symboliques de test, ou une vraie erreur de
    répartition) -- un doublon fausse artificiellement la similarité RBC.
    Ne s'applique qu'aux chemins locaux résolvables sur disque."""
    resolved: dict[Any, list[str]] = {}
    for split, index in (("train", train_index), ("test", test_index)):
        for sid, path in index.items():
            if not isinstance(path, Path):
                continue
            try:
                real = path.resolve()
            except OSError:
                continue
            resolved.setdefault(real, []).append(f"{split}:{sid}")
    duplicates = {p: ks for p, ks in resolved.items() if len(ks) > 1}
    if duplicates:
        msg = (f"[image][ATTENTION] {len(duplicates)} fichier(s) .svs partagé(s) par plusieurs "
               "patientes -- fausse artificiellement la similarité RBC (cas quasi identiques).")
        messages.append(msg)


def load_image_blocks_with_stain_norm(
    train_ids: list[str], y_train: pd.Series,
    test_ids: list[str], y_test: pd.Series,
    train_svs_dir: Path | str | None = None, test_svs_dir: Path | str | None = None,
    train_svs_uploads=None, test_svs_uploads=None,
    thumb_size: tuple[int, int] = (256, 256),
    stain_normalize: bool = True,
    progress_callback: ProgressCB | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Charge et normalise (coloration Reinhard/LAB) les blocs image train et
    test en une seule passe, avec une référence de coloration calculée
    UNIQUEMENT à partir du train (jamais du test, pour éviter toute fuite).
    Retombe sur des caractéristiques synthétiques si openslide est absent,
    qu'aucun dossier/upload n'est fourni, ou qu'aucune lame n'est
    exploitable (comportement inchangé, jamais silencieux : tout est
    consigné dans `info["messages"]`)."""
    info: dict[str, Any] = {"messages": [], "synthetic": False, "n_train_loaded": 0, "n_test_loaded": 0}
    messages = info["messages"]

    train_index: dict[str, Any] = {}
    test_index: dict[str, Any] = {}
    if train_svs_dir:
        train_index.update(_index_svs_files(Path(train_svs_dir)))
    if test_svs_dir:
        test_index.update(_index_svs_files(Path(test_svs_dir)))
    train_index.update(_index_svs_uploads(train_svs_uploads))
    test_index.update(_index_svs_uploads(test_svs_uploads))

    def _synthetic_fallback(msg):
        messages.append(msg)
        info["synthetic"] = True
        _tick(progress_callback, 1.0, "Caractéristiques d'image synthétiques générées.")
        return (generate_synthetic_image_features(train_ids, y_train),
                generate_synthetic_image_features(test_ids, y_test), info)

    if not OPENSLIDE_AVAILABLE:
        return _synthetic_fallback(
            "[image] openslide indisponible -> caractéristiques synthétiques (train+test).")
    if not train_index and not test_index:
        return _synthetic_fallback(
            "[image] aucune lame trouvée (dossier vide ou non fourni) -> "
            "caractéristiques synthétiques (train+test).")

    _check_duplicate_slides(train_index, test_index, messages)

    raw: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}

    def _open_one(path_or_upload) -> Path:
        if isinstance(path_or_upload, Path):
            return path_or_upload
        # objet file-like uploadé : écrire dans un fichier temporaire (openslide
        # a besoin d'un chemin sur disque, pas d'un buffer en mémoire).
        suffix = Path(getattr(path_or_upload, "name", "slide.svs")).suffix or ".svs"
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        try:
            path_or_upload.seek(0)
        except Exception:
            pass
        tmp.write(path_or_upload.read())
        tmp.flush()
        tmp.close()
        return Path(tmp.name)

    def _load_all(index, ids, split):
        kept, missing = [], []
        total = len(ids)
        for n_done, sid in enumerate(ids):
            src = index.get(sid)
            if src is None:
                missing.append(sid)
                continue
            tmp_to_remove = None
            try:
                path = src if isinstance(src, Path) else _open_one(src)
                if not isinstance(src, Path):
                    tmp_to_remove = path
                thumb, mask = _load_thumb_and_mask(path, thumb_size)
                raw[(split, sid)] = (thumb, mask)
                kept.append(sid)
            except Exception as e:
                name = getattr(src, "name", str(src))
                messages.append(f"[image:{split}] ERREUR sur {name} ({sid}) : {e}")
                missing.append(sid)
            finally:
                if tmp_to_remove is not None:
                    try:
                        tmp_to_remove.unlink(missing_ok=True)
                    except Exception:
                        pass
            if total:
                _tick(progress_callback, (n_done + 1) / total / 2.0 + (0.5 if split == "test" else 0.0),
                      f"Chargement des lames {split} ({n_done + 1}/{total})")
        messages.append(f"[image:{split}] {len(kept)}/{len(ids)} lames chargées "
                         f"({len(missing)} manquante(s)/illisible(s)).")
        return kept

    kept_train = _load_all(train_index, train_ids, "train")
    kept_test = _load_all(test_index, test_ids, "test")
    info["n_train_loaded"] = len(kept_train)
    info["n_test_loaded"] = len(kept_test)

    if not kept_train and not kept_test:
        return _synthetic_fallback(
            "[image] aucune lame exploitable -> caractéristiques synthétiques (train+test).")

    target_mean = target_std = None
    if stain_normalize and kept_train:
        means, stds = [], []
        for sid in kept_train:
            thumb, mask = raw[("train", sid)]
            m, s = _lab_stats_tissue(thumb, mask)
            means.append(m)
            stds.append(s)
        target_mean = np.mean(means, axis=0)
        target_std = np.mean(stds, axis=0)
        messages.append(
            f"[image] Normalisation de coloration (Reinhard/LAB), référence = moyenne de "
            f"{len(kept_train)} lames train : L*={target_mean[0]:.1f} a*={target_mean[1]:.1f} "
            f"b*={target_mean[2]:.1f}")
    elif stain_normalize:
        messages.append("[image] Aucune lame train chargée -- normalisation de coloration ignorée.")

    def _build_df(kept, split, ids_all, y):
        rows, kept_ids = [], []
        for sid in kept:
            thumb, mask = raw[(split, sid)]
            if target_mean is not None:
                src_mean, src_std = _lab_stats_tissue(thumb, mask)
                thumb_use = _apply_reinhard(thumb, src_mean, src_std, target_mean, target_std)
            else:
                thumb_use = thumb
            rows.append(_extract_features_from_thumb(thumb_use, mask))
            kept_ids.append(sid)
        if not rows:
            messages.append(f"[image:{split}] aucune lame exploitable -> caractéristiques synthétiques.")
            return generate_synthetic_image_features(ids_all, y)
        return pd.DataFrame(rows, index=kept_ids, columns=IMAGE_FEATURE_NAMES)

    image_train_df = _build_df(kept_train, "train", train_ids, y_train)
    image_test_df = _build_df(kept_test, "test", test_ids, y_test)
    if raw:
        info["_raw_slides"] = {
            sid: raw[(split, sid)]
            for split, sid in raw
        }
        info["_stain_normalize"] = stain_normalize
    return image_train_df, image_test_df, info


def _image_features_for_split(
    raw_slides: dict[str, tuple[np.ndarray, np.ndarray]],
    train_ids: list[str],
    test_ids: list[str],
    stain_normalize: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reconstruit les features avec une référence colorimétrique issue du train."""
    means, stds = [], []
    if stain_normalize:
        for sid in train_ids:
            thumb, mask = raw_slides[sid]
            mean, std = _lab_stats_tissue(thumb, mask)
            means.append(mean)
            stds.append(std)
        target_mean = np.mean(means, axis=0)
        target_std = np.mean(stds, axis=0)
    else:
        target_mean = target_std = None

    def build(ids: list[str]) -> pd.DataFrame:
        rows = []
        for sid in ids:
            thumb, mask = raw_slides[sid]
            if target_mean is not None:
                src_mean, src_std = _lab_stats_tissue(thumb, mask)
                thumb = _apply_reinhard(thumb, src_mean, src_std, target_mean, target_std)
            rows.append(_extract_features_from_thumb(thumb, mask))
        return pd.DataFrame(rows, index=ids, columns=IMAGE_FEATURE_NAMES)

    return build(train_ids), build(test_ids)


# ------------------------------------------------------------------------
# 3) OUTILS DIABLO
# ------------------------------------------------------------------------

def standardize(X, mean=None, std=None):
    X = np.asarray(X, dtype=float)
    if mean is None:
        mean = X.mean(axis=0)
    if std is None:
        std = X.std(axis=0, ddof=1)
        std[std == 0] = 1.0
    return (X - mean) / std, mean, std


def one_hot(y):
    classes = sorted(pd.unique(y))
    Y = np.zeros((len(y), len(classes)))
    for j, c in enumerate(classes):
        Y[:, j] = (np.asarray(y) == c).astype(float)
    return Y, classes


def soft_threshold_topk(w, k):
    if k is None or k >= len(w):
        return w
    order = np.argsort(-np.abs(w))
    keep = order[:k]
    lam = np.abs(w[order[k]]) if k < len(w) else 0.0
    w_new = np.zeros_like(w)
    w_new[keep] = np.sign(w[keep]) * np.maximum(np.abs(w[keep]) - lam, 0.0)
    return w_new


class BlockPLSDA:
    def __init__(self, design, ncomp=2, keepX=None, max_iter=100, tol=1e-6):
        self.design = design
        self.ncomp = ncomp
        self.keepX = keepX
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, blocks_train: dict[str, pd.DataFrame], y_train):
        self.block_names_ = list(blocks_train.keys())
        y_train = pd.Series(np.asarray(y_train))
        self.classes_ = sorted(pd.unique(y_train))

        self.means_, self.stds_ = {}, {}
        X = {}
        for name, df in blocks_train.items():
            Xs, m, s = standardize(df.values)
            X[name] = Xs
            self.means_[name] = m
            self.stds_[name] = s

        Y_dummy, self.classes_ = one_hot(y_train)
        Ys, self.y_mean_, self.y_std_ = standardize(Y_dummy)
        X["Y"] = Ys
        all_blocks = self.block_names_ + ["Y"]

        self.loadings_ = {b: [] for b in self.block_names_}
        self.scores_train_ = {b: [] for b in all_blocks}
        self.var_names_ = {b: list(blocks_train[b].columns) for b in self.block_names_}

        Xcur = {b: X[b].copy() for b in all_blocks}

        for comp in range(self.ncomp):
            t = {}
            for b in all_blocks:
                u, s, vt = np.linalg.svd(Xcur[b], full_matrices=False)
                t[b] = u[:, 0] * s[0]

            a = {b: None for b in all_blocks}
            for _ in range(self.max_iter):
                t_old = {b: t[b].copy() for b in all_blocks}
                for b in all_blocks:
                    acc = np.zeros(Xcur[b].shape[1])
                    for b2 in all_blocks:
                        if b2 == b:
                            continue
                        w = self.design.get(frozenset((b, b2)), 0.0)
                        if w == 0:
                            continue
                        acc += w * (Xcur[b].T @ t[b2])
                    norm = np.linalg.norm(acc)
                    if norm > 0:
                        acc = acc / norm

                    if b != "Y" and self.keepX is not None and b in self.keepX:
                        k = self.keepX[b][comp]
                        acc = soft_threshold_topk(acc, k)
                        norm2 = np.linalg.norm(acc)
                        if norm2 > 0:
                            acc = acc / norm2

                    a[b] = acc
                    t[b] = Xcur[b] @ a[b]

                delta = sum(np.linalg.norm(t[b] - t_old[b]) for b in all_blocks)
                if delta < self.tol:
                    break

            for b in self.block_names_:
                self.loadings_[b].append(a[b])
            for b in all_blocks:
                self.scores_train_[b].append(t[b])

            for b in all_blocks:
                tb = t[b]
                denom = tb @ tb
                if denom <= 1e-12:
                    continue
                p = (Xcur[b].T @ tb) / denom
                Xcur[b] = Xcur[b] - np.outer(tb, p)

        self.centroids_ = {}
        for b in self.block_names_:
            T = np.column_stack(self.scores_train_[b])
            cents = np.zeros((len(self.classes_), self.ncomp))
            for j, c in enumerate(self.classes_):
                mask = (y_train.values == c)
                cents[j] = T[mask].mean(axis=0)
            self.centroids_[b] = cents

        self.block_weights_ = {}
        for b in self.block_names_:
            corrs = [abs(np.corrcoef(self.scores_train_[b][c], self.scores_train_["Y"][c])[0, 1])
                     for c in range(self.ncomp)]
            self.block_weights_[b] = float(np.mean(corrs))

        self.y_train_ = y_train
        return self

    def transform(self, blocks_new: dict[str, pd.DataFrame]):
        scores = {}
        for b, df in blocks_new.items():
            Xs, _, _ = standardize(df.values, self.means_[b], self.stds_[b])
            T = np.column_stack([Xs @ self.loadings_[b][c] for c in range(self.ncomp)])
            scores[b] = T
        return scores

    def predict(self, blocks_new: dict[str, pd.DataFrame], weighted=True):
        scores = self.transform(blocks_new)
        n = next(iter(scores.values())).shape[0]
        votes = np.zeros((n, len(self.classes_)))
        per_block_pred = {}

        for b, T in scores.items():
            cents = self.centroids_[b]
            d = np.linalg.norm(T[:, None, :] - cents[None, :, :], axis=2)
            pred_idx = d.argmin(axis=1)
            per_block_pred[b] = [self.classes_[i] for i in pred_idx]
            w = self.block_weights_[b] if weighted else 1.0
            for i in range(n):
                votes[i, pred_idx[i]] += w

        final_idx = votes.argmax(axis=1)
        final_pred = [self.classes_[i] for i in final_idx]
        return final_pred, per_block_pred


# ------------------------------------------------------------------------
# 3 bis) RBC -- raisonnement à base de cas
# ------------------------------------------------------------------------

def stratified_folds(y: pd.Series, n_folds: int, seed: int = 0) -> list[np.ndarray]:
    """n_folds tableaux d'indices (positions), stratifiés par classe."""
    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(n_folds)]
    for cls in sorted(set(y)):
        idx = np.where(y.values == cls)[0]
        idx = idx.copy()
        rng.shuffle(idx)
        for i, pos in enumerate(idx):
            folds[i % n_folds].append(pos)
    return [np.array(sorted(f)) for f in folds]


class CaseBase:
    """Base de cas multi-blocs pour le RBC. Implémente RETRIEVE (similarité
    pondérée multi-bloc) puis REUSE (vote pondéré), avec :
      - poids de bloc estimés par validation croisée stratifiée répétée
        (plus honnête qu'un LOO brut, notamment pour un bloc bruité comme
        l'image) ;
      - sélection de k (nombre de voisins) par validation croisée ;
      - une évaluation "train" non biaisée (exclusion du cas lui-même)."""

    def __init__(self, blocks: dict[str, pd.DataFrame], y: pd.Series, name: str):
        self.name = name
        self.block_names = list(blocks.keys())
        self.ids = list(blocks[self.block_names[0]].index)
        self.y = pd.Series(list(y.loc[self.ids]) if hasattr(y, "loc") else list(y), index=self.ids)

        self.means_, self.stds_ = {}, {}
        self.Xstd = {}
        self.raw_ = {}
        for b, df in blocks.items():
            Xs, m, s = standardize(df.values)
            self.Xstd[b] = Xs
            self.means_[b] = m
            self.stds_[b] = s
            self.raw_[b] = np.asarray(df.values, dtype=float)

    def compute_block_weights_cv(self, n_folds: int = 5, n_repeats: int = 3, seed: int = 0) -> dict[str, float]:
        """Poids de fiabilité par bloc = exactitude d'un 1-plus-proche-voisin
        (utilisant CE bloc seul) en validation croisée stratifiée répétée."""
        n = len(self.ids)
        n_folds_eff = max(2, min(n_folds, n))
        accum = {b: [] for b in self.block_names}
        for r in range(n_repeats):
            folds = stratified_folds(self.y, n_folds_eff, seed=seed + r)
            for b in self.block_names:
                X = self.Xstd[b]
                correct, total = 0, 0
                for val_idx in folds:
                    val_set = set(val_idx.tolist())
                    ref_idx = np.array([i for i in range(n) if i not in val_set])
                    if len(ref_idx) == 0 or len(val_idx) == 0:
                        continue
                    for i in val_idx:
                        dists = np.linalg.norm(X[ref_idx] - X[i], axis=1)
                        nn_pos = ref_idx[int(np.argmin(dists))]
                        total += 1
                        if self.y.iloc[nn_pos] == self.y.iloc[i]:
                            correct += 1
                accum[b].append(correct / total if total else 0.0)
        return {b: float(np.mean(v)) for b, v in accum.items()}

    def similarity(self, query_blocks: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
        used = [b for b in query_blocks if b in self.Xstd and weights.get(b, 0.0) > 0]
        total_w = sum(weights[b] for b in used)
        sims = np.zeros(len(self.ids))
        if total_w == 0:
            return sims
        for b in used:
            qs, _, _ = standardize(query_blocks[b].reshape(1, -1), self.means_[b], self.stds_[b])
            dists = np.linalg.norm(self.Xstd[b] - qs, axis=1)
            sims += weights[b] * (1.0 / (1.0 + dists))
        return sims / total_w

    def diagnose(self, query_blocks: dict[str, np.ndarray], weights: dict[str, float],
                 k: int = 5, exclude_index: int | None = None):
        sims = self.similarity(query_blocks, weights)
        if exclude_index is not None:
            sims = sims.copy()
            sims[exclude_index] = -np.inf
        k_eff = min(k, len(self.ids) - (1 if exclude_index is not None else 0))
        order = np.argsort(-sims)[:k_eff]
        classes = sorted(set(self.y))
        scores = {c: 0.0 for c in classes}
        for i in order:
            if not np.isfinite(sims[i]):
                continue
            scores[self.y.iloc[i]] += sims[i]
        total = sum(scores.values())
        pred = max(scores, key=scores.get) if total > 0 else classes[0]
        confidence = scores[pred] / total if total > 0 else 0.0
        return pred, confidence

    def retrieve_neighbors(self, query_blocks: dict[str, np.ndarray], weights: dict[str, float],
                            k: int = 5, exclude_index: int | None = None,
                            query_id: str | None = None) -> pd.DataFrame:
        """RETRIEVE du RBC, rendu explicite : renvoie les k plus proches
        voisins d'UNE patiente sous forme de tableau classé (rang, id,
        vraie classe, distance par bloc, score combiné). Réutilise
        EXACTEMENT `self.similarity` -- le même calcul que celui utilisé en
        interne par `diagnose`/`diagnose_batch` pour produire la prédiction
        -- afin de garantir que les voisins listés ici sont bien ceux qui
        pilotent la décision RBC affichée par ailleurs."""
        sims = self.similarity(query_blocks, weights).copy()
        if exclude_index is not None:
            sims[exclude_index] = -np.inf
        k_eff = min(k, len(self.ids) - (1 if exclude_index is not None else 0))
        order = np.argsort(-sims)[:k_eff]

        used = [b for b in query_blocks if b in self.Xstd and weights.get(b, 0.0) > 0]
        rows = []
        rank = 0
        for i in order:
            if not np.isfinite(sims[i]):
                continue
            rank += 1
            row = {
                "rang": rank,
                "patient_id": self.ids[i],
                "classe_vraie": self.y.iloc[i],
                "score_combine": float(sims[i]),
            }
            for b in used:
                qs, _, _ = standardize(np.asarray(query_blocks[b]).reshape(1, -1),
                                        self.means_[b], self.stds_[b])
                row[f"distance_{b}"] = float(np.linalg.norm(self.Xstd[b][i] - qs[0]))
            rows.append(row)

        df = pd.DataFrame(rows)
        if query_id is not None:
            df.insert(0, "query_id", query_id)
        return df

    def _diagnose_restricted(self, query_raw: dict[str, np.ndarray], weights: dict[str, float],
                              ref_idx: np.ndarray, k: int):
        """Prédit en ne considérant que les cas d'indices `ref_idx` comme
        voisins possibles -- utilisé pour la sélection de k par validation
        croisée."""
        sims_full = self.similarity(query_raw, weights)
        sims = np.full(len(self.ids), -np.inf)
        sims[ref_idx] = sims_full[ref_idx]
        k_eff = min(k, len(ref_idx))
        order = np.argsort(-sims)[:k_eff]
        classes = sorted(set(self.y))
        scores = {c: 0.0 for c in classes}
        for i in order:
            if not np.isfinite(sims[i]):
                continue
            scores[self.y.iloc[i]] += sims[i]
        total = sum(scores.values())
        return max(scores, key=scores.get) if total > 0 else classes[0]

    def select_k_cv(self, weights: dict[str, float], k_grid: list[int],
                     n_folds: int = 5, n_repeats: int = 2, seed: int = 10) -> tuple[int, dict[int, float]]:
        n = len(self.ids)
        n_folds_eff = max(2, min(n_folds, n))
        scores = {k: [] for k in k_grid}
        for r in range(n_repeats):
            folds = stratified_folds(self.y, n_folds_eff, seed=seed + r)
            for val_idx in folds:
                val_set = set(val_idx.tolist())
                ref_idx = np.array([i for i in range(n) if i not in val_set])
                if len(ref_idx) == 0 or len(val_idx) == 0:
                    continue
                for k in k_grid:
                    correct = 0
                    for i in val_idx:
                        q = {b: self.raw_[b][i] for b in self.block_names}
                        pred = self._diagnose_restricted(q, weights, ref_idx, k)
                        if pred == self.y.iloc[i]:
                            correct += 1
                    scores[k].append(correct / len(val_idx))
        mean_scores = {k: float(np.mean(v)) if v else 0.0 for k, v in scores.items()}
        best_k = max(mean_scores, key=lambda k: (mean_scores[k], -k))
        return best_k, mean_scores


def diagnose_batch(case_base: CaseBase, weights, query_blocks_dict: dict[str, pd.DataFrame],
                    ids, k=5, exclude_self: bool = False):
    """Diagnostique par RBC un ensemble de patientes. Si `exclude_self` est
    vrai, une patiente présente dans la base de cas (typiquement en train)
    est exclue de sa propre recherche de voisins -- indispensable pour une
    évaluation train non biaisée."""
    id_to_idx = {sid: i for i, sid in enumerate(case_base.ids)}
    preds = []
    for sid in ids:
        q = {b: df.loc[sid].values for b, df in query_blocks_dict.items() if sid in df.index}
        exclude_idx = id_to_idx.get(sid) if exclude_self else None
        pred, _ = case_base.diagnose(q, weights, k=k, exclude_index=exclude_idx)
        preds.append(pred)
    return preds


def retrieve_neighbors_batch(case_base: CaseBase, weights, query_blocks_dict: dict[str, pd.DataFrame],
                              ids, k: int = 5, exclude_self: bool = False) -> dict[str, pd.DataFrame]:
    """Comme `diagnose_batch`, mais pour la citation des voisins (RETRIEVE) :
    renvoie, pour chaque patiente de `ids`, le tableau de ses k plus proches
    voisines dans la base de cas (voir `CaseBase.retrieve_neighbors`)."""
    id_to_idx = {sid: i for i, sid in enumerate(case_base.ids)}
    out: dict[str, pd.DataFrame] = {}
    for sid in ids:
        q = {b: df.loc[sid].values for b, df in query_blocks_dict.items() if sid in df.index}
        exclude_idx = id_to_idx.get(sid) if exclude_self else None
        out[sid] = case_base.retrieve_neighbors(q, weights, k=k, exclude_index=exclude_idx, query_id=sid)
    return out


# ------------------------------------------------------------------------
# 4) MÉTRIQUES
# ------------------------------------------------------------------------

def compute_confusion_matrix(y_true, y_pred, classes):
    classes = list(classes)
    idx = {c: i for i, c in enumerate(classes)}
    cm = np.zeros((len(classes), len(classes)), dtype=int)
    for yt, yp in zip(y_true, y_pred):
        cm[idx[yt], idx[yp]] += 1
    return cm


def compute_metrics(y_true, y_pred, classes, include_macro_precision_recall: bool = True):
    classes = list(classes)
    cm = compute_confusion_matrix(y_true, y_pred, classes)
    acc = float(np.trace(cm) / cm.sum()) if cm.sum() else 0.0

    per_class = []
    for i, cls in enumerate(classes):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum() - tp)
        fn = int(cm[i, :].sum() - tp)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_class.append({"precision": precision, "recall": recall, "f1_score": f1})

    metrics = {
        "accuracy": acc,
        "f1_macro": float(np.mean([p["f1_score"] for p in per_class])) if per_class else 0.0,
    }
    if include_macro_precision_recall:
        metrics["precision_macro"] = (
            float(np.mean([p["precision"] for p in per_class])) if per_class else 0.0
        )
        metrics["recall_macro"] = (
            float(np.mean([p["recall"] for p in per_class])) if per_class else 0.0
        )
    return metrics, cm


def accuracy(pred, truth):
    return float(np.mean(np.asarray(pred) == np.asarray(truth)))


def print_confusion_like_mixomics(cm: np.ndarray, classes: list[str]) -> str:
    """Formate la matrice de confusion dans le même format que la sortie R de
    mixOmics (get.confusion_matrix / perf.block.splsda), pour comparaison
    directe ligne à ligne avec le tutoriel officiel. Retourne le texte
    (au lieu de l'imprimer) pour un affichage souple côté UI."""
    col_labels = [f"predicted.as.{c}" for c in classes]
    col_w = [max(len(lbl), 6) for lbl in col_labels]
    row_label_w = max(len(c) for c in classes)
    header = " " * row_label_w + "  " + "  ".join(f"{lbl:>{w}}" for lbl, w in zip(col_labels, col_w))
    lines = ["  " + header]
    for i, cls in enumerate(classes):
        row = "  ".join(f"{cm[i, j]:>{col_w[j]}d}" for j in range(len(classes)))
        lines.append(f"  {cls:<{row_label_w}}  {row}")
    return "\n".join(lines)


# ------------------------------------------------------------------------
# 5) STRUCTURES DE RÉSULTATS + ORCHESTRATION
# ------------------------------------------------------------------------

@dataclass
class ModelParams:
    design_between: float = 0.1
    design_toward_y: float = 1.0  # fixé dans le script d'origine (jamais exposé comme réglable)
    ncomp: int = 2
    keepX_mrna: list[int] = field(default_factory=lambda: [8, 5])
    keepX_mirna: list[int] = field(default_factory=lambda: [10, 5])
    keepX_protein: list[int] = field(default_factory=lambda: [6, 5])
    keepX_image: list[int] = field(default_factory=lambda: [8, 6])


@dataclass
class RBCParams:
    k_grid: list[int] = field(default_factory=lambda: [3, 5, 7, 9, 11, 15, 21])
    n_folds_weights: int = 5
    n_repeats_weights: int = 3
    n_folds_k: int = 5
    n_repeats_k: int = 2
    seed_weights_omics: int = 0
    seed_weights_full: int = 1
    seed_k_omics: int = 10
    seed_k_full: int = 11


@dataclass
class PipelineResults:
    classes_all: list[str]
    reference_rows: list[dict]
    reference_metrics: pd.DataFrame
    reference_confusion_matrices: dict[str, dict[str, np.ndarray]]
    results_detail: pd.DataFrame
    comparison_summary: pd.DataFrame
    confusion_matrices: dict[tuple[str, str, str], np.ndarray]
    weights_omics: dict[str, float]
    k_omics: int
    k_scores_omics: dict[int, float]
    weights_full: dict[str, float]
    k_full: int | None
    k_scores_full: dict[int, float]
    common_train: list[str]
    common_test: list[str]
    n_train: int
    n_test: int
    image_info: dict
    messages: list[str]
    csv_exports: dict[str, pd.DataFrame]
    mixomics_text_reference_test: str
    mixomics_text_reference_test_model: dict[str, str]
    data_preview: dict = field(default_factory=dict)
    neighbors_test: dict[tuple[str, str], dict] = field(default_factory=dict)
    include_image: bool = True
    repeated_comparison_summary: pd.DataFrame = field(default_factory=pd.DataFrame)


MIN_PATIENTS_PAR_CLASSE = 2


def run_pipeline(
    train: dict[str, pd.DataFrame], y_train: pd.Series,
    test: dict[str, pd.DataFrame], y_test: pd.Series,
    image_train: pd.DataFrame, image_test: pd.DataFrame,
    image_info: dict,
    model_params: ModelParams | None = None,
    rbc_params: RBCParams | None = None,
    progress_callback: ProgressCB | None = None,
    include_image: bool = True,
    include_macro_precision_recall: bool = True,
) -> PipelineResults:
    """Orchestration complète : DIABLO dense/sparse (omique seule, et
    omique+image si `include_image=True`), RBC dans l'espace latent,
    évaluation de référence (150 train / 70 test complètes, comparable au
    tutoriel officiel mixOmics -- omique seule, TOUJOURS calculée, image ou
    pas) et comparaison équitable (sous-ensemble commun avec image,
    UNIQUEMENT si `include_image=True`). Quand `include_image=False`
    (case "Ignorer les images" cochée dans l'interface), aucun modèle
    omique+image n'est construit et aucun résultat comparatif A/B n'est
    calculé : seule l'analyse omique seule est produite. N'écrit rien sur
    disque -- tout est renvoyé dans un `PipelineResults`."""
    mp = model_params or ModelParams()
    rp = rbc_params or RBCParams()
    messages: list[str] = list(image_info.get("messages", []))

    classes_all = sorted(set(y_train))

    # ---- Aperçu des données réellement chargées (diagnostic "quelles
    # données ont vraiment servi ?", visible dans l'onglet Résumé) --------
    data_preview: dict[str, Any] = {
        "blocks_train": {b: {"n_patients": len(df), "sample_ids": list(df.index[:5])}
                          for b, df in train.items()},
        "blocks_test": {b: {"n_patients": len(df), "sample_ids": list(df.index[:5])}
                         for b, df in test.items()},
        "blocks_used": sorted(set(train.keys()) | {"image"}),
        "class_distribution_train": y_train.value_counts().to_dict(),
        "class_distribution_test": y_test.value_counts().to_dict(),
    }
    data_preview["blocks_train"]["image"] = {
        "n_patients": len(image_train), "sample_ids": list(image_train.index[:5])}
    data_preview["blocks_test"]["image"] = {
        "n_patients": len(image_test), "sample_ids": list(image_test.index[:5])}

    # Blocs omiques réellement présents (mrna+mirna toujours requis, protein
    # optionnel -- voir `load_breast_tcga_from_uploads`). Rien ci-dessous ne
    # suppose un nombre fixe de blocs : le design et keepX sont construits
    # dynamiquement à partir de `omics_block_names`.
    omics_block_names = list(train.keys())
    if len(omics_block_names) < 2:
        raise ValueError(
            "Il faut au moins 2 blocs omiques (mrna + mirna) pour construire "
            "un design multi-blocs DIABLO."
        )

    _tick(progress_callback, 0.05, "Construction des modèles DIABLO (omique seule)...")

    keepX_map = {"mrna": mp.keepX_mrna, "mirna": mp.keepX_mirna, "protein": mp.keepX_protein}

    design_omics: dict[frozenset, float] = {}
    for i, b1 in enumerate(omics_block_names):
        for b2 in omics_block_names[i + 1:]:
            design_omics[frozenset((b1, b2))] = mp.design_between
        design_omics[frozenset((b1, "Y"))] = mp.design_toward_y
    keepX_omics = {b: keepX_map[b] for b in omics_block_names if b in keepX_map}

    plsda_omics = BlockPLSDA(design=design_omics, ncomp=mp.ncomp, keepX=None).fit(train, y_train)
    splsda_omics = BlockPLSDA(design=design_omics, ncomp=mp.ncomp, keepX=keepX_omics).fit(train, y_train)

    common_train: list[str] = []
    common_test: list[str] = []
    train_full: dict[str, pd.DataFrame] = {}
    test_full: dict[str, pd.DataFrame] = {}
    y_train_full = y_train.iloc[0:0]
    y_test_full = y_test.iloc[0:0]
    plsda_full = splsda_full = None

    if not include_image:
        messages.append(
            "[image] Analyse omique seule demandée (\"Ignorer les images\") -- "
            "aucun modèle omique+image construit, aucune comparaison A/B "
            "calculée. Seule l'évaluation de référence (omique seule) est "
            "produite."
        )
    else:
        _tick(progress_callback, 0.15, "Construction des modèles DIABLO (omique + image)...")

        common_train = [i for i in train["mrna"].index if i in image_train.index]
        common_test = [i for i in test["mrna"].index if i in image_test.index]

        n_classes_train = y_train.loc[common_train].nunique() if common_train else 0
        classes_ok = (
            n_classes_train >= 2
            and y_train.loc[common_train].value_counts().min() >= MIN_PATIENTS_PAR_CLASSE
        ) if common_train else False
        skip_image_model = len(common_train) < 6 or len(common_test) < 1 or not classes_ok
        if skip_image_model:
            raise ValueError(
                f"Trop peu de patientes avec une image exploitable ({len(common_train)} train / "
                f"{len(common_test)} test, {n_classes_train} classe(s) représentée(s)) pour "
                "construire une comparaison omique vs omique+image fiable. Cochez "
                "\"Ignorer les images\" pour une analyse omique seule."
            )

        train_full = {b: df.loc[common_train] for b, df in train.items()}
        train_full["image"] = image_train.loc[common_train]
        y_train_full = y_train.loc[common_train]

        test_full = {b: df.loc[common_test] for b, df in test.items()}
        test_full["image"] = image_test.loc[common_test]
        y_test_full = y_test.loc[common_test]

        design_full = dict(design_omics)
        for b in omics_block_names:
            design_full[frozenset((b, "image"))] = mp.design_between
        design_full[frozenset(("image", "Y"))] = mp.design_toward_y
        keepX_full = dict(keepX_omics)
        keepX_full["image"] = mp.keepX_image

        plsda_full = BlockPLSDA(design=design_full, ncomp=mp.ncomp, keepX=None).fit(train_full, y_train_full)
        splsda_full = BlockPLSDA(design=design_full, ncomp=mp.ncomp, keepX=keepX_full).fit(train_full, y_train_full)

    _tick(progress_callback, 0.35, "RBC : projection dans l'espace latent DIABLO...")

    def _latent_blocks(model: BlockPLSDA, blocks: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        scores = model.transform(blocks)
        out = {}
        for b, T in scores.items():
            cols = [f"{b}_comp{i + 1}" for i in range(T.shape[1])]
            out[b] = pd.DataFrame(T, index=blocks[b].index, columns=cols)
        return out

    latent_train_omics = _latent_blocks(plsda_omics, train)
    latent_test_omics = _latent_blocks(plsda_omics, test)

    _tick(progress_callback, 0.45, "RBC : poids de bloc et sélection de k (validation croisée)...")

    case_base_omics = CaseBase(latent_train_omics, y_train, name="omique_seule_latent")
    weights_omics = case_base_omics.compute_block_weights_cv(
        n_folds=rp.n_folds_weights, n_repeats=rp.n_repeats_weights, seed=rp.seed_weights_omics)
    k_omics, k_scores_omics = case_base_omics.select_k_cv(
        weights_omics, k_grid=rp.k_grid, n_folds=rp.n_folds_k, n_repeats=rp.n_repeats_k,
        seed=rp.seed_k_omics)
    messages.append(f"Poids des blocs (omique seule, CV, espace latent) : "
                     f"{ {b: round(w, 3) for b, w in weights_omics.items()} }")
    messages.append(f"k sélectionné (omique seule) : {k_omics}")

    latent_train_full: dict[str, pd.DataFrame] = {}
    latent_test_full: dict[str, pd.DataFrame] = {}
    case_base_full: CaseBase | None = None
    weights_full: dict[str, float] = {}
    k_full: int | None = None
    k_scores_full: dict[int, float] = {}

    if include_image:
        latent_train_full = _latent_blocks(plsda_full, train_full)
        latent_test_full = _latent_blocks(plsda_full, test_full)

        case_base_full = CaseBase(latent_train_full, y_train_full, name="omique_image_latent")
        weights_full = case_base_full.compute_block_weights_cv(
            n_folds=rp.n_folds_weights, n_repeats=rp.n_repeats_weights, seed=rp.seed_weights_full)
        k_full, k_scores_full = case_base_full.select_k_cv(
            weights_full, k_grid=rp.k_grid, n_folds=rp.n_folds_k, n_repeats=rp.n_repeats_k,
            seed=rp.seed_k_full)
        messages.append(f"Poids des blocs (omique+image, CV, espace latent) : "
                         f"{ {b: round(w, 3) for b, w in weights_full.items()} }")
        messages.append(f"k sélectionné (omique+image) : {k_full}")

    # ---- RBC -- citation des plus proches voisines (test) -----------------
    # Précalculé pour TOUTES les patientes test (jeu raisonnable, <= ~100
    # patientes typiquement), config A sur le test complet (pas besoin
    # d'image), config B sur le sous-ensemble commun (image requise).
    _tick(progress_callback, 0.5, "RBC : citation des plus proches voisines (test)...")

    def _neighbors_with_prediction(cbase, cweights, k_sel, blocks_dict, ids, y_true):
        out: dict[str, dict] = {}
        for sid, df_nn in retrieve_neighbors_batch(
            cbase, cweights, blocks_dict, ids, k=k_sel, exclude_self=False,
        ).items():
            q = {b: df.loc[sid].values for b, df in blocks_dict.items() if sid in df.index}
            pred, confidence = cbase.diagnose(q, cweights, k=k_sel, exclude_index=None)
            true_label = y_true.loc[sid] if sid in y_true.index else None
            out[sid] = {
                "table": df_nn, "true_label": true_label,
                "predicted_label": pred, "confidence": confidence,
                "agree": (true_label == pred) if true_label is not None else None,
            }
        return out

    neighbors_test: dict[tuple[str, str], dict] = {}
    for sid, info in _neighbors_with_prediction(
        case_base_omics, weights_omics, k_omics, latent_test_omics,
        list(test["mrna"].index), y_test,
    ).items():
        neighbors_test[("A) omique seule", sid)] = info
    if include_image:
        for sid, info in _neighbors_with_prediction(
            case_base_full, weights_full, k_full, latent_test_full,
            common_test, y_test,
        ).items():
            neighbors_test[("B) omique+image", sid)] = info

    # ---- 5 bis) RÉFÉRENCE -- omique seule, TOUT le jeu ---------------------
    _tick(progress_callback, 0.55, "Évaluation de référence (comparable au tutoriel mixOmics)...")

    reference_confusion_matrices: dict[str, dict[str, np.ndarray]] = {}
    mixomics_text_by_model: dict[str, str] = {}

    def report_reference(model, model_type):
        pred_train, _ = model.predict(train)
        pred_test, _ = model.predict(test)
        m_train, cm_train = compute_metrics(
            list(y_train), pred_train, classes_all, include_macro_precision_recall
        )
        m_test, cm_test = compute_metrics(
            list(y_test), pred_test, classes_all, include_macro_precision_recall
        )
        reference_confusion_matrices[model_type] = {"train": cm_train, "test": cm_test}
        mixomics_text_by_model[model_type] = print_confusion_like_mixomics(cm_test, classes_all)
        return m_train, m_test

    ref_dense_train, ref_dense_test = report_reference(plsda_omics, "dense")
    ref_sparse_train, ref_sparse_test = report_reference(splsda_omics, "sparse")

    pred_train_rbc_ref = diagnose_batch(case_base_omics, weights_omics, latent_train_omics,
                                         list(train["mrna"].index), k=k_omics, exclude_self=True)
    pred_test_rbc_ref = diagnose_batch(case_base_omics, weights_omics, latent_test_omics,
                                        list(test["mrna"].index), k=k_omics, exclude_self=False)
    ref_rbc_train, cm_rbc_train = compute_metrics(
        list(y_train), pred_train_rbc_ref, classes_all, include_macro_precision_recall
    )
    ref_rbc_test, cm_rbc_test = compute_metrics(
        list(y_test), pred_test_rbc_ref, classes_all, include_macro_precision_recall
    )
    reference_confusion_matrices["rbc"] = {"train": cm_rbc_train, "test": cm_rbc_test}
    mixomics_text_by_model["rbc"] = print_confusion_like_mixomics(cm_rbc_test, classes_all)

    reference_rows = [
        {"model_type": "dense", "split": "train", "n_patients": len(y_train), **ref_dense_train},
        {"model_type": "dense", "split": "test", "n_patients": len(y_test), **ref_dense_test},
        {"model_type": "sparse", "split": "train", "n_patients": len(y_train), **ref_sparse_train},
        {"model_type": "sparse", "split": "test", "n_patients": len(y_test), **ref_sparse_test},
        {"model_type": "rbc", "split": "train", "n_patients": len(y_train), **ref_rbc_train},
        {"model_type": "rbc", "split": "test", "n_patients": len(y_test), **ref_rbc_test},
    ]
    reference_metrics = pd.DataFrame(reference_rows)

    reference_confusion_rows = []
    for model_type, splits in reference_confusion_matrices.items():
        for split_name, cm in splits.items():
            for i, true_cls in enumerate(classes_all):
                for j, pred_cls in enumerate(classes_all):
                    reference_confusion_rows.append({
                        "model_type": model_type, "split": split_name,
                        "classe_vraie": true_cls, "classe_predite": pred_cls, "n": int(cm[i, j]),
                    })
    reference_confusion_df = pd.DataFrame(reference_confusion_rows)

    # ---- 6) COMPARAISON ÉQUITABLE (UNIQUEMENT si include_image=True) -------
    rows: list[dict] = []
    confusion_matrices: dict[tuple[str, str, str], np.ndarray] = {}
    comparison_columns = [
        "model_type", "split", "n_patients",
        "accuracy_omics", "accuracy_omics_image", "delta_accuracy",
        "f1_macro_omics", "f1_macro_omics_image", "delta_f1_macro",
    ]
    if include_macro_precision_recall:
        comparison_columns.extend([
            "precision_macro_omics", "precision_macro_omics_image",
            "recall_macro_omics", "recall_macro_omics_image",
        ])
    comparison_df = pd.DataFrame(columns=comparison_columns)

    if include_image:
        _tick(progress_callback, 0.75, "Comparaison équitable (sous-ensemble commun avec image)...")

        train_omics_common = {b: df.loc[common_train] for b, df in train.items()}
        test_omics_common = {b: df.loc[common_test] for b, df in test.items()}

        diablo_configs = [
            ("dense", "A) omique seule", plsda_omics, train_omics_common, y_train_full, test_omics_common, y_test_full),
            ("dense", "B) omique+image", plsda_full, train_full, y_train_full, test_full, y_test_full),
            ("sparse", "A) omique seule", splsda_omics, train_omics_common, y_train_full, test_omics_common, y_test_full),
            ("sparse", "B) omique+image", splsda_full, train_full, y_train_full, test_full, y_test_full),
        ]
        for model_type, config_name, model, tr_blocks, tr_y, te_blocks, te_y in diablo_configs:
            for split_name, blocks, y_true in [("train", tr_blocks, tr_y), ("test", te_blocks, te_y)]:
                pred, _ = model.predict(blocks)
                metrics, cm = compute_metrics(
                    list(y_true), pred, classes_all, include_macro_precision_recall
                )
                confusion_matrices[(model_type, config_name, split_name)] = cm
                rows.append({"model_type": model_type, "config": config_name, "split": split_name,
                              "n_patients": len(y_true), **metrics})

        latent_train_omics_common = {b: df.loc[common_train] for b, df in latent_train_omics.items()}
        latent_test_omics_common = {b: df.loc[common_test] for b, df in latent_test_omics.items()}

        rbc_configs = [
            ("A) omique seule", case_base_omics, weights_omics, k_omics,
             latent_train_omics_common, y_train_full, latent_test_omics_common, y_test_full),
            ("B) omique+image", case_base_full, weights_full, k_full,
             latent_train_full, y_train_full, latent_test_full, y_test_full),
        ]
        for config_name, cbase, cweights, k_sel, tr_blocks, tr_y, te_blocks, te_y in rbc_configs:
            for split_name, blocks, y_true, ids, excl in [
                ("train", tr_blocks, tr_y, list(tr_y.index), True),
                ("test", te_blocks, te_y, list(te_y.index), False),
            ]:
                pred = diagnose_batch(cbase, cweights, blocks, ids, k=k_sel, exclude_self=excl)
                metrics, cm = compute_metrics(
                    list(y_true), pred, classes_all, include_macro_precision_recall
                )
                confusion_matrices[("rbc", config_name, split_name)] = cm
                rows.append({"model_type": "rbc", "config": config_name, "split": split_name,
                              "n_patients": len(y_true), **metrics})

        results_df = pd.DataFrame(rows)

        comparison_rows = []
        for model_type in ["dense", "sparse", "rbc"]:
            for split_name in ["train", "test"]:
                row_a = results_df[(results_df.model_type == model_type) & (results_df.split == split_name)
                                    & (results_df.config == "A) omique seule")].iloc[0]
                row_b = results_df[(results_df.model_type == model_type) & (results_df.split == split_name)
                                    & (results_df.config == "B) omique+image")].iloc[0]
                comparison_row = {
                    "model_type": model_type,
                    "split": split_name,
                    "n_patients": row_a["n_patients"],
                    "accuracy_omics": row_a["accuracy"],
                    "accuracy_omics_image": row_b["accuracy"],
                    "delta_accuracy": row_b["accuracy"] - row_a["accuracy"],
                    "f1_macro_omics": row_a["f1_macro"],
                    "f1_macro_omics_image": row_b["f1_macro"],
                    "delta_f1_macro": row_b["f1_macro"] - row_a["f1_macro"],
                }
                if include_macro_precision_recall:
                    comparison_row.update({
                        "precision_macro_omics": row_a["precision_macro"],
                        "precision_macro_omics_image": row_b["precision_macro"],
                        "recall_macro_omics": row_a["recall_macro"],
                        "recall_macro_omics_image": row_b["recall_macro"],
                    })
                comparison_rows.append(comparison_row)
        comparison_df = pd.DataFrame(comparison_rows)
    else:
        results_df = pd.DataFrame(columns=["model_type", "config", "split", "n_patients"])

    confusion_rows = []
    for (model_type, config_name, split_name), cm in confusion_matrices.items():
        for i, true_cls in enumerate(classes_all):
            for j, pred_cls in enumerate(classes_all):
                confusion_rows.append({
                    "model_type": model_type, "config": config_name, "split": split_name,
                    "classe_vraie": true_cls, "classe_predite": pred_cls, "n": int(cm[i, j]),
                })
    confusion_df = pd.DataFrame(confusion_rows)

    _tick(progress_callback, 1.0, "Terminé.")

    def _neighbors_long_df(config_name: str, y_query: pd.Series) -> pd.DataFrame:
        rows_nn = []
        for (cfg, qid), info in neighbors_test.items():
            df_nn = info["table"]
            if cfg != config_name or df_nn.empty:
                continue
            qy = y_query.loc[qid] if qid in y_query.index else None
            for _, r in df_nn.iterrows():
                rows_nn.append({
                    "query_id": qid,
                    "query_true_label": qy,
                    "rank": int(r["rang"]),
                    "neighbor_id": r["patient_id"],
                    "neighbor_true_label": r["classe_vraie"],
                    "score": float(r["score_combine"]),
                })
        return pd.DataFrame(rows_nn)

    neighbors_omics_df = _neighbors_long_df("A) omique seule", y_test)
    neighbors_full_df = _neighbors_long_df("B) omique+image", y_test_full)

    csv_exports = {
        "reference_omics_only_full.csv": reference_metrics,
        "comparison_summary.csv": comparison_df,
        "results_detail.csv": results_df,
        "confusion_matrices.csv": confusion_df,
        "reference_confusion_matrices.csv": reference_confusion_df,
        "rbc_nearest_neighbors_omics.csv": neighbors_omics_df,
        "rbc_nearest_neighbors_full.csv": neighbors_full_df,
    }

    return PipelineResults(
        classes_all=classes_all,
        reference_rows=reference_rows,
        reference_metrics=reference_metrics,
        reference_confusion_matrices=reference_confusion_matrices,
        results_detail=results_df,
        comparison_summary=comparison_df,
        confusion_matrices=confusion_matrices,
        weights_omics=weights_omics,
        k_omics=k_omics,
        k_scores_omics=k_scores_omics,
        weights_full=weights_full,
        k_full=k_full,
        k_scores_full=k_scores_full,
        common_train=common_train,
        common_test=common_test,
        n_train=len(y_train),
        n_test=len(y_test),
        image_info=image_info,
        messages=messages,
        csv_exports=csv_exports,
        mixomics_text_reference_test=mixomics_text_by_model.get("dense", ""),
        mixomics_text_reference_test_model=mixomics_text_by_model,
        data_preview=data_preview,
        neighbors_test=neighbors_test,
        include_image=include_image,
    )


def run_repeated_stratified_comparison(
    train: dict[str, pd.DataFrame],
    y_train: pd.Series,
    test: dict[str, pd.DataFrame],
    y_test: pd.Series,
    image_train: pd.DataFrame,
    image_test: pd.DataFrame,
    image_info: dict,
    model_params: ModelParams,
    rbc_params: RBCParams,
    include_image: bool = True,
    repetitions: int = REPEATED_N_REPEATS,
    test_size: int = REPEATED_TEST_SIZE,
    seed: int = REPEATED_SEED,
    progress_callback: ProgressCB | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare all models on paired, stratified 150/70 splits of all patients."""
    if repetitions < 2:
        raise ValueError("Il faut au moins 2 répétitions pour calculer un écart-type.")
    if test_size != REPEATED_TEST_SIZE:
        raise ValueError(
            f"L'étude répétée utilise un test fixe de {REPEATED_TEST_SIZE} patientes."
        )

    cohort_y = pd.concat([y_train, y_test])
    if cohort_y.index.has_duplicates:
        raise ValueError("Les identifiants patientes doivent être uniques entre train et test.")
    if len(cohort_y) != REPEATED_TOTAL_PATIENTS:
        raise ValueError(
            f"L'étude répétée attend {REPEATED_TOTAL_PATIENTS} patientes au total "
            f"({REPEATED_TOTAL_PATIENTS - test_size} train / {test_size} test), "
            f"mais {len(cohort_y)} ont été fournies."
        )
    if test_size >= len(cohort_y):
        raise ValueError("La taille du test doit être inférieure à la cohorte totale.")

    omics_block_names = [name for name in train if name != "protein"]
    missing_test_blocks = [name for name in omics_block_names if name not in test]
    if missing_test_blocks:
        raise ValueError(
            "Les blocs omiques suivants doivent être disponibles pour toutes les "
            "patientes de l'étude répétée : " + ", ".join(missing_test_blocks) + "."
        )
    if len(omics_block_names) < 2:
        raise ValueError(
            "L'étude répétée nécessite au moins deux blocs omiques communs "
            "(mRNA et miRNA)."
        )
    cohort_blocks = {
        name: pd.concat([train[name], test[name]]).loc[cohort_y.index]
        for name in omics_block_names
    }

    cohort_images = pd.concat([image_train, image_test])
    if cohort_images.index.has_duplicates:
        raise ValueError("Les identifiants des blocs image doivent être uniques entre train et test.")
    if include_image:
        missing_images = cohort_y.index.difference(cohort_images.index)
        if len(missing_images):
            raise ValueError(
                f"La comparaison répétée appariée nécessite des descripteurs image pour "
                f"les {REPEATED_TOTAL_PATIENTS} patientes ; {len(missing_images)} en "
                "manquent. Complétez les images ou cochez « Ignorer les images »."
            )
        cohort_images = cohort_images.loc[cohort_y.index]

    raw_slides = image_info.get("_raw_slides")
    if include_image and raw_slides is not None:
        missing_slides = cohort_y.index.difference(pd.Index(raw_slides))
        if len(missing_slides):
            raise ValueError(
                "La normalisation Reinhard par partition exige une lame exploitable "
                f"pour chaque patiente ; {len(missing_slides)} lame(s) manquante(s)."
            )

    splitter = StratifiedShuffleSplit(
        n_splits=repetitions,
        test_size=test_size,
        random_state=seed,
    )
    split_indices = list(
        splitter.split(np.zeros((len(cohort_y), 1)), cohort_y.to_numpy())
    )

    repeated_model_params = ModelParams(
        design_between=0.1,
        design_toward_y=1.0,
        ncomp=2,
        keepX_mrna=list(model_params.keepX_mrna),
        keepX_mirna=list(model_params.keepX_mirna),
        keepX_protein=list(model_params.keepX_protein),
        keepX_image=list(model_params.keepX_image),
    )
    repeated_rbc_params = RBCParams(
        k_grid=list(rbc_params.k_grid),
        n_folds_weights=5,
        n_repeats_weights=2,
        n_folds_k=5,
        n_repeats_k=2,
        seed_weights_omics=rbc_params.seed_weights_omics,
        seed_weights_full=rbc_params.seed_weights_full,
        seed_k_omics=rbc_params.seed_k_omics,
        seed_k_full=rbc_params.seed_k_full,
    )
    metrics = [
        "accuracy_omics", "accuracy_omics_image", "delta_accuracy",
        "f1_macro_omics", "f1_macro_omics_image", "delta_f1_macro",
    ]

    run_rows: list[dict] = []
    for repetition, (train_idx, test_idx) in enumerate(split_indices, start=1):
        fold_train_ids = cohort_y.index[train_idx].tolist()
        fold_test_ids = cohort_y.index[test_idx].tolist()
        fold_train = {
            name: block.loc[fold_train_ids] for name, block in cohort_blocks.items()
        }
        fold_test = {
            name: block.loc[fold_test_ids] for name, block in cohort_blocks.items()
        }
        fold_y_train = cohort_y.loc[fold_train_ids]
        fold_y_test = cohort_y.loc[fold_test_ids]
        if include_image and raw_slides is not None:
            fold_image_train, fold_image_test = _image_features_for_split(
                raw_slides,
                fold_train_ids,
                fold_test_ids,
                stain_normalize=bool(image_info.get("_stain_normalize", True)),
            )
        else:
            fold_image_train = cohort_images.loc[fold_train_ids]
            fold_image_test = cohort_images.loc[fold_test_ids]

        fold_results = run_pipeline(
            fold_train,
            fold_y_train,
            fold_test,
            fold_y_test,
            fold_image_train,
            fold_image_test,
            {"messages": []},
            model_params=repeated_model_params,
            rbc_params=repeated_rbc_params,
            include_image=include_image,
            include_macro_precision_recall=False,
        )

        for model_type in ("dense", "sparse", "rbc"):
            for split_name in ("train", "test"):
                record = {
                    "repetition": repetition,
                    "model_type": model_type,
                    "split": split_name,
                    "n_patients": len(fold_train_ids if split_name == "train" else fold_test_ids),
                    **{metric: np.nan for metric in metrics},
                }
                if include_image:
                    match = fold_results.comparison_summary[
                        (fold_results.comparison_summary["model_type"] == model_type)
                        & (fold_results.comparison_summary["split"] == split_name)
                    ]
                    if match.empty:
                        raise ValueError(
                            f"Résultat manquant pour le modèle « {model_type} », "
                            f"split « {split_name} », répétition {repetition}."
                        )
                    for metric in metrics:
                        record[metric] = float(match.iloc[0][metric])
                else:
                    match = fold_results.reference_metrics[
                        (fold_results.reference_metrics["model_type"] == model_type)
                        & (fold_results.reference_metrics["split"] == split_name)
                    ]
                    if match.empty:
                        raise ValueError(
                            f"Résultat manquant pour le modèle « {model_type} », "
                            f"split « {split_name} », répétition {repetition}."
                        )
                    record["accuracy_omics"] = float(match.iloc[0]["accuracy"])
                    record["f1_macro_omics"] = float(match.iloc[0]["f1_macro"])
                run_rows.append(record)

        if progress_callback is not None:
            progress_callback(
                repetition / repetitions,
                f"Évaluation stratifiée : répétition {repetition}/{repetitions} "
                f"({len(fold_train_ids)} entraînement / {len(fold_test_ids)} test)",
            )

    runs = pd.DataFrame(
        run_rows,
        columns=["repetition", "model_type", "split", "n_patients", *metrics],
    )
    summary_rows = []
    for (model_type, split_name), model_runs in runs.groupby(
        ["model_type", "split"], sort=False
    ):
        summary = {
            "model_type": model_type,
            "split": split_name,
            "n_patients": int(round(model_runs["n_patients"].mean())),
            "repetitions": int(model_runs["repetition"].nunique()),
        }
        for metric in metrics:
            values = model_runs[metric].dropna()
            summary[f"{metric}_mean"] = float(values.mean()) if not values.empty else np.nan
            summary[f"{metric}_std"] = (
                float(values.std(ddof=1)) if len(values) > 1 else np.nan
            )
        summary_rows.append(summary)

    return pd.DataFrame(summary_rows), runs


# ------------------------------------------------------------------------
# 7) FIGURES (retournent une Figure matplotlib, aucun savefig/close interne)
# ------------------------------------------------------------------------

def plot_comparison(comparison_df: pd.DataFrame, metric_col_a: str, metric_col_b: str,
                     metric_label: str, title: str) -> plt.Figure | None:
    if comparison_df.empty:
        return None
    model_types = [m for m in ["dense", "sparse", "rbc"] if m in comparison_df.model_type.unique()]
    if not model_types:
        return None
    splits = ["train", "test"]

    fig, axes = plt.subplots(1, len(model_types), figsize=(5 * len(model_types), 5), sharey=True)
    if len(model_types) == 1:
        axes = [axes]
    bar_width = 0.32
    x = np.arange(len(splits))

    for ax, model_type in zip(axes, model_types):
        sub = comparison_df[comparison_df.model_type == model_type].set_index("split").loc[splits]
        vals_a = sub[metric_col_a].values
        vals_b = sub[metric_col_b].values

        bars_a = ax.bar(x - bar_width / 2, vals_a, bar_width, label="A) omique seule", color=COLOR_OMICS)
        bars_b = ax.bar(x + bar_width / 2, vals_b, bar_width, label="B) omique + image", color=COLOR_OMICS_IMAGE)

        for bars in (bars_a, bars_b):
            for rect in bars:
                height = rect.get_height()
                ax.annotate(f"{height:.1%}", xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points",
                            ha="center", va="bottom", fontsize=9)

        ax.set_xticks(x)
        ax.set_xticklabels(["Train", "Test"])
        ax.set_title(MODEL_TYPE_LABELS.get(model_type, model_type), fontsize=11)
        ax.set_ylim(0, 1.08)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", color="#E5E5E5", lw=0.8, zorder=0)
        ax.set_axisbelow(True)

    axes[0].set_ylabel(metric_label)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.04), frameon=False)
    fig.suptitle(title, y=1.10, fontsize=13)
    fig.tight_layout()
    return fig


def plot_confusion_matrices(confusion_matrices: dict, classes_all: list[str],
                             split_name: str, title: str) -> plt.Figure | None:
    model_types = [m for m in ["dense", "sparse", "rbc"] if any(
        (m, cfg, split_name) in confusion_matrices for cfg in CONFIG_ORDER)]
    if not model_types:
        return None

    n_rows, n_cols = len(CONFIG_ORDER), len(model_types)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.6 * n_cols, 4.6 * n_rows))
    if n_rows == 1:
        axes = axes[None, :]
    if n_cols == 1:
        axes = axes[:, None]

    n_classes = len(classes_all)
    vmax = max((cm.max() for cm in confusion_matrices.values() if cm.size), default=1)

    im = None
    for r, config_name in enumerate(CONFIG_ORDER):
        for c, model_type in enumerate(model_types):
            ax = axes[r, c]
            cm = confusion_matrices.get((model_type, config_name, split_name))
            if cm is None:
                ax.axis("off")
                continue
            im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=vmax, aspect="equal")

            row_sums = cm.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            frac = cm / row_sums
            for i in range(n_classes):
                for j in range(n_classes):
                    val = cm[i, j]
                    txt_color = "white" if frac[i, j] > 0.6 else "#1a1a1a"
                    ax.text(j, i, str(int(val)), ha="center", va="center",
                            fontsize=10, color=txt_color)

            ax.set_xticks(range(n_classes))
            ax.set_yticks(range(n_classes))
            ax.set_xticklabels(classes_all, rotation=45, ha="right", fontsize=8)
            ax.set_yticklabels(classes_all, fontsize=8)
            if r == n_rows - 1:
                ax.set_xlabel("Prédit", fontsize=9)
            if c == 0:
                ax.set_ylabel(f"{CONFIG_SHORT[config_name]}\nVrai", fontsize=9)
            acc = np.trace(cm) / cm.sum() if cm.sum() else 0.0
            ax.set_title(f"{MODEL_TYPE_LABELS.get(model_type, model_type)}\n(exactitude {acc:.1%})",
                          fontsize=10)
            for spine in ax.spines.values():
                spine.set_visible(False)

    fig.tight_layout(rect=[0, 0, 0.90, 0.93])
    if im is not None:
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        fig.colorbar(im, cax=cbar_ax, label="Nombre de patientes")
    fig.suptitle(title, fontsize=13, y=0.98)
    return fig


def plot_reference_confusion_matrices(reference_confusion_matrices: dict, classes_all: list[str],
                                       split_name: str, title: str) -> plt.Figure | None:
    """Matrices de RÉFÉRENCE (150 train / 70 test COMPLÈTES, omique seule) --
    directement comparables, ligne à ligne, à la matrice de confusion du cas
    d'usage officiel mixOmics (https://mixomics.org/mixdiablo/diablo-tcga-case-study/)."""
    model_types = [m for m in ["dense", "sparse", "rbc"]
                   if split_name in reference_confusion_matrices.get(m, {})]
    if not model_types:
        return None

    fig, axes = plt.subplots(1, len(model_types), figsize=(4.6 * len(model_types), 4.8))
    if len(model_types) == 1:
        axes = [axes]

    n_classes = len(classes_all)
    vmax = max(cm.max() for cm in
               [reference_confusion_matrices[m][split_name] for m in model_types])

    im = None
    for ax, model_type in zip(axes, model_types):
        cm = reference_confusion_matrices[model_type][split_name]
        im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=vmax, aspect="equal")
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        frac = cm / row_sums
        for i in range(n_classes):
            for j in range(n_classes):
                txt_color = "white" if frac[i, j] > 0.6 else "#1a1a1a"
                ax.text(j, i, str(int(cm[i, j])), ha="center", va="center",
                        fontsize=11, color=txt_color)
        ax.set_xticks(range(n_classes))
        ax.set_yticks(range(n_classes))
        ax.set_xticklabels(classes_all, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(classes_all, fontsize=9)
        ax.set_xlabel("Prédit", fontsize=9)
        ax.set_ylabel("Vrai", fontsize=9)
        acc = np.trace(cm) / cm.sum() if cm.sum() else 0.0
        ax.set_title(f"{MODEL_TYPE_LABELS.get(model_type, model_type)}\n(exactitude {acc:.1%}, n={int(cm.sum())})",
                      fontsize=10)
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.tight_layout(rect=[0, 0, 0.90, 0.90])
    if im is not None:
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        fig.colorbar(im, cax=cbar_ax, label="Nombre de patientes")
    fig.suptitle(title, fontsize=13, y=0.98)
    return fig
