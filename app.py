"""
================================================================================
 Application Streamlit -- DIABLO + RBC : intégration multi-omique et
 histopathologie
================================================================================
Interface web pour le pipeline diablo_core : comparaison DIABLO omique seule
vs. omique + image, en dense/sparse, plus un classifieur RBC (raisonnement à
base de cas) opérant dans l'espace latent de DIABLO.
================================================================================
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pandas as pd
import streamlit as st

import diablo_core as dc

st.set_page_config(
    page_title="DIABLO + RBC",
    page_icon="🧬",
    layout="wide",
)

st.title("DIABLO + RBC — Intégration multi-omique et histopathologie")
st.caption(
    "Comparaison omique seule vs. omique + image (DIABLO dense/sparse) et "
    "raisonnement à base de cas (RBC) dans l'espace latent DIABLO."
)


# ------------------------------------------------------------------------
# Barre latérale -- données et paramètres
# ------------------------------------------------------------------------

with st.sidebar:
    st.header("1. Données omiques")
    data_mode = st.radio(
        "Source des données",
        ["Données synthétiques de démonstration", "Mes propres données"],
        index=0,
    )

    uploaded_omics = {}
    synth_params = {"n_train": 150, "n_test": 70, "seed": 42}

    if data_mode == "Données synthétiques de démonstration":
        st.caption(
            "Jeu synthétique imitant le cas d'usage TCGA-BRCA de mixOmics "
            "(3 sous-types : Basal, Her2, LumA). Le bloc protéines n'existe "
            "qu'en train, comme dans le tutoriel officiel."
        )
        synth_params["n_train"] = st.number_input("Nombre de patientes (train)", 30, 1000, 150, step=10)
        synth_params["n_test"] = st.number_input("Nombre de patientes (test)", 10, 500, 70, step=10)
        synth_params["seed"] = st.number_input("Graine aléatoire (seed)", 0, 10_000, 42)
    else:
        st.caption(
            "Formats attendus : CSV avec identifiants patientes en première "
            "colonne (index), puis les variables. Le fichier `subtype` "
            "contient une colonne `subtype` (ou toute colonne unique) avec "
            "le sous-type de chaque patiente. Les identifiants sont "
            "réalignés strictement par ID entre tous les fichiers (jamais "
            "par simple correspondance de nombre de lignes) : toute "
            "patiente absente d'un fichier est écartée et signalée, jamais "
            "mal associée silencieusement."
        )
        uploaded_omics["train_mrna"] = st.file_uploader("data.train_mrna.csv (requis)", type="csv")
        uploaded_omics["train_mirna"] = st.file_uploader("data.train_mirna.csv (requis)", type="csv")
        uploaded_omics["train_subtype"] = st.file_uploader("data.train_subtype.csv (requis)", type="csv")
        uploaded_omics["test_mrna"] = st.file_uploader("data.test_mrna.csv (requis)", type="csv")
        uploaded_omics["test_mirna"] = st.file_uploader("data.test_mirna.csv (requis)", type="csv")
        uploaded_omics["test_subtype"] = st.file_uploader("data.test_subtype.csv (requis)", type="csv")
        uploaded_omics["train_protein"] = st.file_uploader(
            "data.train_protein.csv (optionnel -- s'il y en a)", type="csv")
        uploaded_omics["test_protein"] = st.file_uploader(
            "data.test_protein.csv (optionnel -- s'il y en a)", type="csv")
        st.info(
            "Le bloc protéines est **optionnel**, côté train comme côté "
            "test : s'il n'est fourni pour aucun des deux, l'analyse tourne "
            "avec mRNA + miRNA (et l'image, si fournie). Dans le tutoriel "
            "officiel mixOmics TCGA-BRCA, ce bloc n'existe qu'en train -- "
            "mais si vos propres données en ont aussi côté test, vous "
            "pouvez le fournir ici."
        )

    st.header("2. Données image (histopathologie)")
    st.caption(
        "Optionnel : s'il n'y a pas de lames disponibles, cochez "
        "\"Ignorer les images\" -- l'analyse tourne alors en omique seule "
        "(seule la config A du comparatif est vraiment exploitée). Si des "
        "lames sont fournies mais que certaines patientes n'en ont pas, "
        "l'application le gère normalement : les lames manquantes sont "
        "comptabilisées et affichées (X/Y lames chargées), sans faire "
        "échouer l'analyse."
    )
    skip_images = st.checkbox("Ignorer les images (analyse omique seule)", value=False)

    image_source_mode = st.radio(
        "Source des lames .svs",
        ["Dossier local (chemin)", "Téléverser des fichiers .svs"],
        index=0,
        disabled=skip_images,
    )

    train_svs_dir = test_svs_dir = None
    train_svs_uploads = test_svs_uploads = None

    if not skip_images:
        if image_source_mode == "Dossier local (chemin)":
            st.caption(
                "Utile si l'application tourne sur votre propre machine : "
                "évite de téléverser de gros fichiers .svs."
            )
            train_svs_dir = st.text_input("Dossier des lames train (.svs)", value="")
            test_svs_dir = st.text_input("Dossier des lames test (.svs)", value="")
        else:
            train_svs_uploads = st.file_uploader(
                "Lames train (.svs)", type=["svs"], accept_multiple_files=True)
            test_svs_uploads = st.file_uploader(
                "Lames test (.svs)", type=["svs"], accept_multiple_files=True)

        stain_normalize = st.checkbox("Normalisation de coloration (Reinhard)", value=True)
        thumb_w = st.number_input("Taille de vignette (largeur, px)", 32, 1024, 256, step=32)
        thumb_h = st.number_input("Taille de vignette (hauteur, px)", 32, 1024, 256, step=32)
    else:
        stain_normalize = True
        thumb_w, thumb_h = 256, 256

    if not dc.OPENSLIDE_AVAILABLE:
        st.warning(
            "openslide n'est pas installé dans cet environnement : les "
            "caractéristiques d'image seront **synthétiques** quelles que "
            "soient les lames fournies."
        )

    st.header("3. Paramètres du modèle DIABLO")
    design_between = st.number_input(
        "Poids de design entre blocs (mrna/mirna/protein/image)", 0.0, 1.0, 0.1, step=0.05)
    st.caption("Poids de design vers Y : fixé à 1.0 (valeur du script d'origine, non réglable).")
    ncomp = st.number_input("Nombre de composantes (ncomp)", 1, 5, 2)

    st.subheader("keepX (modèle sparse)")
    col1, col2 = st.columns(2)
    with col1:
        keepX_mrna_1 = st.number_input("mRNA, comp. 1", 1, 500, 8)
        keepX_mirna_1 = st.number_input("miRNA, comp. 1", 1, 500, 10)
        keepX_protein_1 = st.number_input("Protéines, comp. 1", 1, 500, 6)
        keepX_image_1 = st.number_input("Image, comp. 1", 1, 12, 8)
    with col2:
        keepX_mrna_2 = st.number_input("mRNA, comp. 2", 1, 500, 5)
        keepX_mirna_2 = st.number_input("miRNA, comp. 2", 1, 500, 5)
        keepX_protein_2 = st.number_input("Protéines, comp. 2", 1, 500, 5)
        keepX_image_2 = st.number_input("Image, comp. 2", 1, 12, 6)

    st.header("4. Paramètres RBC")
    k_grid_str = st.text_input("Grille de k (voisins), séparés par des virgules",
                                value="3,5,7,9,11,15,21")
    n_folds_weights = st.number_input("CV -- plis (poids de bloc)", 2, 20, 5)
    n_repeats_weights = st.number_input("CV -- répétitions (poids de bloc)", 1, 20, 3)
    n_folds_k = st.number_input("CV -- plis (sélection de k)", 2, 20, 5)
    n_repeats_k = st.number_input("CV -- répétitions (sélection de k)", 1, 20, 2)
    st.caption("L'étude répétée fixe la CV RBC à 5 plis × 2 répétitions.")

    st.header("5. Étude répétée")
    repeated_repetitions = st.number_input(
        "Nombre de partitions stratifiées",
        min_value=2,
        max_value=100,
        value=dc.REPEATED_N_REPEATS,
        step=1,
    )
    st.caption(
        "Étude sur les 220 patientes : 150 entraînement / 70 test à chaque "
        "partition (seed 0). Le RBC utilise une CV 5 plis × 2 répétitions."
    )

    st.divider()
    run_clicked = st.button("🚀 Lancer l'analyse", type="primary", use_container_width=True)


# ------------------------------------------------------------------------
# Exécution du pipeline
# ------------------------------------------------------------------------

def _build_run_key():
    """Construit une clé simple pour savoir si les paramètres ont changé,
    afin que le bouton déclenche toujours une exécution réelle quand les
    entrées changent."""
    key = (
        data_mode, tuple(sorted(synth_params.items())),
        tuple((k, getattr(v, "name", None), getattr(v, "size", None)) for k, v in uploaded_omics.items()),
        skip_images, image_source_mode, train_svs_dir, test_svs_dir,
        tuple(getattr(f, "name", None) for f in (train_svs_uploads or [])),
        tuple(getattr(f, "name", None) for f in (test_svs_uploads or [])),
        stain_normalize, thumb_w, thumb_h,
        design_between, ncomp,
        keepX_mrna_1, keepX_mrna_2, keepX_mirna_1, keepX_mirna_2,
        keepX_protein_1, keepX_protein_2, keepX_image_1, keepX_image_2,
        k_grid_str, n_folds_weights, n_repeats_weights, n_folds_k, n_repeats_k,
    )
    return key


if run_clicked:
    try:
        with st.status("Exécution du pipeline DIABLO + RBC...", expanded=True) as status:
            progress_bar = st.progress(0.0)

            def progress_callback(frac, msg):
                progress_bar.progress(frac)
                status.update(label=msg)

            # ---- 1) chargement des données omiques -------------------------
            status.write("Chargement des données omiques...")
            loader_messages: list[str] = []
            if data_mode == "Données synthétiques de démonstration":
                train, y_train, test, y_test = dc.generate_synthetic_breast_tcga(
                    n_train=int(synth_params["n_train"]),
                    n_test=int(synth_params["n_test"]),
                    seed=int(synth_params["seed"]),
                )
            else:
                # mRNA + miRNA sont le socle indispensable ; la protéine est
                # optionnelle (train uniquement, s'il y en a).
                required = ["train_mrna", "train_mirna", "train_subtype",
                            "test_mrna", "test_mirna", "test_subtype"]
                missing = [k for k in required if uploaded_omics.get(k) is None]
                if missing:
                    raise ValueError(
                        "Fichiers CSV manquants : " + ", ".join(missing) +
                        ". Veuillez téléverser au minimum mRNA, miRNA et "
                        "subtype pour le train et le test (la protéine est "
                        "optionnelle), ou utiliser les données synthétiques "
                        "de démonstration."
                    )
                train, y_train, test, y_test = dc.load_breast_tcga_from_uploads(
                    uploaded_omics["train_mrna"], uploaded_omics["train_mirna"],
                    uploaded_omics["train_subtype"],
                    uploaded_omics["test_mrna"], uploaded_omics["test_mirna"],
                    uploaded_omics["test_subtype"],
                    train_protein=uploaded_omics.get("train_protein"),
                    test_protein=uploaded_omics.get("test_protein"),
                    messages=loader_messages,
                )
                for m in loader_messages:
                    status.write(m)
            status.write(f"Train : {len(y_train)} patientes -- Test : {len(y_test)} patientes.")
            status.write(f"Blocs omiques utilisés : {', '.join(train.keys())}"
                         + (" (protéine incluse)" if "protein" in train else " (pas de bloc protéine)"))

            # ---- 2) bloc image ----------------------------------------------
            status.write("Construction du bloc image (normalisation de coloration incluse)...")
            if skip_images:
                image_train = dc.generate_synthetic_image_features(
                    list(train["mrna"].index), y_train)
                image_test = dc.generate_synthetic_image_features(
                    list(test["mrna"].index), y_test)
                image_info = {
                    "messages": ["[image] Analyse omique seule demandée -- "
                                 "bloc image synthétique généré mais non utilisé "
                                 "pour la comparaison équitable."],
                    "synthetic": True, "n_train_loaded": 0, "n_test_loaded": 0,
                }
            else:
                image_train, image_test, image_info = dc.load_image_blocks_with_stain_norm(
                    list(train["mrna"].index), y_train,
                    list(test["mrna"].index), y_test,
                    train_svs_dir=train_svs_dir or None,
                    test_svs_dir=test_svs_dir or None,
                    train_svs_uploads=train_svs_uploads,
                    test_svs_uploads=test_svs_uploads,
                    thumb_size=(int(thumb_w), int(thumb_h)),
                    stain_normalize=stain_normalize,
                )
            # les messages d'alignement des étiquettes (partie la plus
            # sensible pour "des résultats différents") sont placés en tête
            # du journal, avant les messages du bloc image.
            image_info["messages"] = loader_messages + list(image_info.get("messages", []))
            for m in image_info.get("messages", []):
                status.write(m)
            if not skip_images and not image_info.get("synthetic"):
                status.write(
                    f"Lames exploitables : {image_info.get('n_train_loaded', 0)}/{len(y_train)} "
                    f"train, {image_info.get('n_test_loaded', 0)}/{len(y_test)} test "
                    "(couverture partielle gérée normalement : les patientes sans lame "
                    "sont simplement exclues de la comparaison omique+image)."
                )
            if image_info.get("synthetic"):
                st.info(
                    "Caractéristiques d'image **synthétiques** utilisées (pas de lame "
                    "exploitable, openslide indisponible, ou analyse omique seule "
                    "demandée). Le pipeline s'exécute normalement de bout en bout."
                )

            # ---- 2 bis) aperçu des données réellement chargées --------------
            with st.expander("Aperçu des données chargées (avant lancement du pipeline)", expanded=True):
                st.caption(
                    "Utile pour vérifier d'un coup d'œil que ce sont bien VOS "
                    "données (et non le jeu synthétique) qui ont été utilisées."
                )
                preview_rows = []
                for split_name, blocks in [("train", train), ("test", test)]:
                    for b, df in blocks.items():
                        preview_rows.append({
                            "split": split_name, "bloc": b, "n_patientes": len(df),
                            "premiers ID": ", ".join(map(str, df.index[:5])),
                        })
                st.dataframe(pd.DataFrame(preview_rows), use_container_width=True)
                st.caption(
                    f"Blocs omiques utilisés : **{', '.join(train.keys())}**"
                    + (" -- protéine incluse." if "protein" in train else " -- pas de bloc protéine.")
                )
                col_prev1, col_prev2 = st.columns(2)
                with col_prev1:
                    st.caption("Distribution des classes (train)")
                    st.dataframe(y_train.value_counts().rename("n_patientes"), use_container_width=True)
                with col_prev2:
                    st.caption("Distribution des classes (test)")
                    st.dataframe(y_test.value_counts().rename("n_patientes"), use_container_width=True)

            # ---- 3) pipeline complet -----------------------------------------
            k_grid = [int(x.strip()) for x in k_grid_str.split(",") if x.strip()]
            model_params = dc.ModelParams(
                design_between=float(design_between),
                ncomp=int(ncomp),
                keepX_mrna=[int(keepX_mrna_1), int(keepX_mrna_2)],
                keepX_mirna=[int(keepX_mirna_1), int(keepX_mirna_2)],
                keepX_protein=[int(keepX_protein_1), int(keepX_protein_2)],
                keepX_image=[int(keepX_image_1), int(keepX_image_2)],
            )
            rbc_params = dc.RBCParams(
                k_grid=k_grid,
                n_folds_weights=int(n_folds_weights),
                n_repeats_weights=int(n_repeats_weights),
                n_folds_k=int(n_folds_k),
                n_repeats_k=int(n_repeats_k),
            )

            results = dc.run_pipeline(
                train, y_train, test, y_test,
                image_train, image_test, image_info,
                model_params=model_params, rbc_params=rbc_params,
                progress_callback=progress_callback,
                include_image=not skip_images,
            )

            status.update(
                label=f"Calcul de l'étude comparative sur {int(repeated_repetitions)} partitions...",
                state="running",
                expanded=True,
            )
            repeated_progress = st.progress(0.0)

            def repeated_progress_callback(frac, msg):
                repeated_progress.progress(frac)
                status.update(label=msg)

            repeated_summary, repeated_runs = dc.run_repeated_stratified_comparison(
                train,
                y_train,
                test,
                y_test,
                image_train,
                image_test,
                image_info,
                model_params=model_params,
                rbc_params=rbc_params,
                include_image=not skip_images,
                repetitions=int(repeated_repetitions),
                progress_callback=repeated_progress_callback,
            )
            results.repeated_comparison_summary = repeated_summary
            results.csv_exports["repeated_comparison_summary.csv"] = repeated_summary
            results.csv_exports["repeated_comparison_runs.csv"] = repeated_runs

            st.session_state["pipeline_results"] = results
            st.session_state["pipeline_messages"] = results.messages
            status.update(label="Analyse et étude comparative terminées.", state="complete")
    except Exception as e:
        st.error(f"Erreur pendant l'exécution du pipeline : {e}")


# ------------------------------------------------------------------------
# Résultats
# ------------------------------------------------------------------------

results: dc.PipelineResults | None = st.session_state.get("pipeline_results")

if results is None:
    st.info(
        "Configurez les données et les paramètres dans la barre latérale, "
        "puis cliquez sur **Lancer l'analyse**."
    )
else:
    tabs = st.tabs([
        "Étude de référence : DIABLO vs RBC (omiques)",
        "Matrices de confusion — Étude de référence",
        "Étude comparative : DIABLO vs RBC (omique + images)",
        "Matrices de confusion — Étude comparative",
        "RBC — détails",
        "Étude répétée : DIABLO et RBC",
        "Téléchargements",
    ])

    # ---- Résumé ----------------------------------------------------------
    with tabs[0]:
        st.subheader("Référence (150 train / 70 test complètes, comparable au tutoriel mixOmics)")
        st.caption(
            "Exactitude / F1 par modèle (dense/sparse/RBC), omique seule, sur "
            "l'ensemble complet train/test -- directement comparable au "
            "tutoriel officiel mixOmics."
        )
        st.dataframe(results.reference_metrics, use_container_width=True)

        if results.include_image:
            st.caption(
                "Le détail par configuration (A/B) et le delta omique vs. "
                "omique+image sont dans l'onglet **Étude comparative : DIABLO "
                "vs RBC (omique + images)** -- pas répété ici pour éviter le "
                "doublon avec ce tableau "
                "(ses lignes « A) omique seule » y sont identiques dès lors "
                "que toutes les patientes ont une image)."
            )

        st.subheader("Aperçu des données utilisées pour cette exécution")
        st.caption(
            "Permet de vérifier après coup ce qui a réellement été utilisé "
            "(données synthétiques ou réelles, blocs présents, alignement "
            "des étiquettes)."
        )
        dp = results.data_preview
        if dp:
            st.write(f"Blocs en jeu : **{', '.join(dp.get('blocks_used', []))}**")
            preview_rows = []
            for split_name, blocks_key in [("train", "blocks_train"), ("test", "blocks_test")]:
                for b, info in dp.get(blocks_key, {}).items():
                    preview_rows.append({
                        "split": split_name, "bloc": b, "n_patientes": info["n_patients"],
                        "premiers ID": ", ".join(map(str, info["sample_ids"])),
                    })
            st.dataframe(pd.DataFrame(preview_rows), use_container_width=True)
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                st.caption("Distribution des classes (train)")
                st.dataframe(
                    pd.Series(dp.get("class_distribution_train", {}), name="n_patientes"),
                    use_container_width=True)
            with col_d2:
                st.caption("Distribution des classes (test)")
                st.dataframe(
                    pd.Series(dp.get("class_distribution_test", {}), name="n_patientes"),
                    use_container_width=True)

    # ---- Matrices de confusion -- référence -------------------------------
    with tabs[1]:
        st.caption(
            "Matrices de confusion **de référence**, calculées sur les 150 "
            "patientes train / 70 patientes test complètes (omique seule), "
            "**directement comparables** à celles du cas d'usage officiel "
            "mixOmics TCGA-BRCA (mêmes 70 patientes test)."
        )
        fig_ref_test = dc.plot_reference_confusion_matrices(
            results.reference_confusion_matrices, results.classes_all, "test",
            "Matrices de confusion RÉFÉRENCE (70 patientes test, omique seule)\n"
            "-- comparable au cas d'usage officiel mixOmics")
        if fig_ref_test is not None:
            st.pyplot(fig_ref_test)

        fig_ref_train = dc.plot_reference_confusion_matrices(
            results.reference_confusion_matrices, results.classes_all, "train",
            "Matrices de confusion RÉFÉRENCE (150 patientes train, omique seule)")
        if fig_ref_train is not None:
            st.pyplot(fig_ref_train)

        with st.expander("Format texte façon mixOmics (test, modèle dense)"):
            st.code(results.mixomics_text_reference_test_model.get("dense", ""), language=None)

    # ---- Matrices de confusion -- comparaison équitable --------------------
    with tabs[3]:
        if not results.include_image:
            st.info(
                "« Ignorer les images » était coché pour cette exécution : "
                "seule l'analyse omique seule a été calculée, il n'y a pas de "
                "comparaison A/B à afficher ici. Voir l'onglet **Matrices de "
                "confusion — Étude de référence** pour les résultats "
                "omique seule."
            )
        else:
            st.caption(
                "Matrices de confusion de la **comparaison équitable** : calculées "
                f"sur le sous-ensemble commun de {len(results.common_train)} patientes "
                f"train / {len(results.common_test)} patientes test disposant d'une "
                "image exploitable, pour la config A (omique seule) et la config B "
                "(omique + image). **Ces chiffres ne sont PAS directement "
                "comparables** à ceux du tutoriel officiel mixOmics (sous-ensemble "
                "de patientes différent des 150/70 complètes)."
            )
            fig_cmp_test = dc.plot_confusion_matrices(
                results.confusion_matrices, results.classes_all, "test",
                "Matrices de confusion (test) -- par approche")
            if fig_cmp_test is not None:
                st.pyplot(fig_cmp_test)

            fig_cmp_train = dc.plot_confusion_matrices(
                results.confusion_matrices, results.classes_all, "train",
                "Matrices de confusion (train) -- par approche")
            if fig_cmp_train is not None:
                st.pyplot(fig_cmp_train)

    # ---- Comparaison équitable (barres) ------------------------------------
    with tabs[2]:
        if not results.include_image:
            st.info(
                "« Ignorer les images » était coché pour cette exécution : "
                "il n'y a pas de config omique+image à comparer à l'omique "
                "seule. Décochez cette case et relancez l'analyse pour "
                "obtenir cette comparaison."
            )
        else:
            st.caption(
                "Effet du bloc image, à patientes strictement identiques : "
                "config A (omique seule) vs. config B (omique + image)."
            )
            st.dataframe(results.comparison_summary, use_container_width=True)

            fig_acc = dc.plot_comparison(
                results.comparison_summary, "accuracy_omics", "accuracy_omics_image",
                "Exactitude (accuracy)", "Comparaison des performances -- Exactitude")
            if fig_acc is not None:
                st.pyplot(fig_acc)

            fig_f1 = dc.plot_comparison(
                results.comparison_summary, "f1_macro_omics", "f1_macro_omics_image",
                "F1-score (macro)", "Comparaison des performances -- F1-score macro")
            if fig_f1 is not None:
                st.pyplot(fig_f1)

    # ---- RBC -- détails ------------------------------------------------------
    with tabs[4]:
        if results.include_image:
            col1, col2 = st.columns(2)
        else:
            col1, = st.columns(1)
        with col1:
            st.subheader("Config A -- omique seule")
            st.metric("k sélectionné", results.k_omics)
            st.bar_chart(pd.Series(results.weights_omics, name="poids"))
            st.caption("Score de validation croisée par valeur de k :")
            st.dataframe(
                pd.DataFrame({"k": list(results.k_scores_omics.keys()),
                              "accuracy_cv": list(results.k_scores_omics.values())}),
                use_container_width=True,
            )
        if results.include_image:
            with col2:
                st.subheader("Config B -- omique + image")
                st.metric("k sélectionné", results.k_full)
                st.bar_chart(pd.Series(results.weights_full, name="poids"))
                st.caption("Score de validation croisée par valeur de k :")
                st.dataframe(
                    pd.DataFrame({"k": list(results.k_scores_full.keys()),
                                  "accuracy_cv": list(results.k_scores_full.values())}),
                    use_container_width=True,
                )
        else:
            st.info(
                "Config B (omique + image) non calculée -- « Ignorer les "
                "images » était coché pour cette exécution."
            )

        st.divider()
        st.subheader("Citer les patientes les plus proches (RETRIEVE du RBC)")
        st.caption(
            "Pour une patiente test donnée, voici exactement les k plus "
            "proches voisines utilisées par le RBC pour construire sa "
            "prédiction -- le calcul de similarité est rigoureusement le "
            "même que celui qui produit la prédiction affichée ailleurs "
            "dans cet onglet."
        )

        configs_available = [c for c in ["A) omique seule", "B) omique+image"]
                              if any(cfg == c for cfg, _ in results.neighbors_test.keys())]
        if not configs_available:
            st.info("Aucun voisinage RBC disponible pour cette exécution.")
        else:
            col_sel1, col_sel2 = st.columns([1, 2])
            with col_sel1:
                nn_config = st.radio("Configuration", configs_available, key="nn_config")
            patient_ids = sorted({sid for cfg, sid in results.neighbors_test.keys() if cfg == nn_config})
            with col_sel2:
                nn_patient = st.selectbox(
                    "Patiente test", patient_ids, index=0 if patient_ids else None, key="nn_patient")

            if nn_patient is not None:
                info = results.neighbors_test.get((nn_config, nn_patient))
                if info is None or info["table"].empty:
                    st.warning("Pas de voisin exploitable pour cette patiente.")
                else:
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Vrai sous-type", str(info["true_label"]))
                    m2.metric("Sous-type prédit (RBC)", str(info["predicted_label"]))
                    m3.metric("Accord vrai/prédit", "Oui" if info["agree"] else "Non")
                    st.caption(f"Confiance RBC (part du score des voisins pour la classe prédite) : "
                               f"{info['confidence']:.1%}")
                    display_cols = ["rang", "patient_id", "classe_vraie", "score_combine"]
                    extra_cols = [c for c in info["table"].columns if c.startswith("distance_")]
                    st.dataframe(
                        info["table"][display_cols + extra_cols].rename(columns={
                            "patient_id": "ID voisine", "classe_vraie": "vraie classe",
                            "score_combine": "score combiné (similarité pondérée)",
                        }),
                        use_container_width=True, hide_index=True,
                    )

        with st.expander("Journal d'exécution"):
            for m in results.messages:
                st.text(m)

        st.info(
            "**Rappel méthodologique** : le RBC ne retrouve pas ses cas voisins "
            "dans les variables omiques brutes, mais dans les scores latents à "
            "2 dimensions par bloc appris par block.plsda (dense). Il n'est donc "
            "plus un algorithme totalement indépendant de DIABLO -- c'est un "
            "raisonnement à base de cas appliqué à la représentation apprise par "
            "DIABLO. La comparaison RBC vs DIABLO doit être lue comme deux "
            "stratégies de décision sur la même représentation, pas comme deux "
            "méthodes indépendantes."
        )

    # ---- Comparaison répétée et écarts-types ------------------------------
    with tabs[5]:
        st.subheader("Étude répétée : DIABLO dense, DIABLO sparse et RBC")
        st.caption(
            "À chaque répétition, une partition StratifiedShuffleSplit est tirée "
            "sur l'ensemble des 220 patientes (150 entraînement / 70 test), "
            "stratifiée sur le sous-type. Les mêmes patientes sont utilisées "
            "pour les trois modèles et les configurations A/B. Tous les "
            "prétraitements, DIABLO et réglages RBC sont réajustés sur le train "
            "de la partition ; le test sert uniquement à l'évaluation de cette "
            "répétition. Les protéines sont exclues de cette étude répétée."
        )
        st.dataframe(
            results.repeated_comparison_summary,
            hide_index=True,
        )
        st.caption(
            "Chaque métrique est affichée sous forme de moyenne et d'écart-type "
            "d'échantillon (ddof=1). Les deltas B − A sont calculés séparément "
            "à chaque partition avant leur agrégation. Si les images sont "
            "ignorées, seules les métriques A sont renseignées."
        )

    # ---- Téléchargements -------------------------------------------------
    with tabs[6]:
        st.write(
            "Télécharger l'ensemble des résultats (CSV + figures PNG) sous "
            "forme d'archive zip."
        )

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for fname, df in results.csv_exports.items():
                zf.writestr(fname, df.to_csv(index=False))

            figures = [
                ("confusion_matrices_reference_test.png",
                 dc.plot_reference_confusion_matrices(
                     results.reference_confusion_matrices, results.classes_all, "test",
                     "Matrices de confusion RÉFÉRENCE (70 patientes test, omique seule)\n"
                     "-- comparable au cas d'usage officiel mixOmics")),
                ("confusion_matrices_reference_train.png",
                 dc.plot_reference_confusion_matrices(
                     results.reference_confusion_matrices, results.classes_all, "train",
                     "Matrices de confusion RÉFÉRENCE (150 patientes train, omique seule)")),
                ("confusion_matrices_test.png",
                 dc.plot_confusion_matrices(
                     results.confusion_matrices, results.classes_all, "test",
                     "Matrices de confusion (test) -- par approche")),
                ("confusion_matrices_train.png",
                 dc.plot_confusion_matrices(
                     results.confusion_matrices, results.classes_all, "train",
                     "Matrices de confusion (train) -- par approche")),
                ("comparison_accuracy.png",
                 dc.plot_comparison(
                     results.comparison_summary, "accuracy_omics", "accuracy_omics_image",
                     "Exactitude (accuracy)", "Comparaison des performances -- Exactitude")),
                ("comparison_f1.png",
                 dc.plot_comparison(
                     results.comparison_summary, "f1_macro_omics", "f1_macro_omics_image",
                     "F1-score (macro)", "Comparaison des performances -- F1-score macro")),
            ]
            for fname, fig in figures:
                if fig is None:
                    continue
                img_buf = io.BytesIO()
                fig.savefig(img_buf, format="png", dpi=200, bbox_inches="tight")
                zf.writestr(fname, img_buf.getvalue())

        st.download_button(
            "⬇️ Télécharger tous les résultats (.zip)",
            data=buf.getvalue(),
            file_name="comparaison_omics_vs_image.zip",
            mime="application/zip",
            use_container_width=True,
        )
