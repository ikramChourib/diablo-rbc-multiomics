# DIABLO + RBC — Application Streamlit

Application web pour le pipeline de comparaison **DIABLO** (multi-block
sPLS-DA, réimplémentation from-scratch façon mixOmics) **omique seule vs.
omique + histopathologie**, avec un classifieur **RBC** (raisonnement à base
de cas, k-NN) opérant dans l'espace latent appris par DIABLO.

## Installation

```bash
pip install -r requirements.txt
```

`openslide-python` est optionnel (commenté dans `requirements.txt`) : il ne
sert qu'à lire de vraies lames `.svs`. Sans lui (ou sans lame fournie),
l'application utilise automatiquement des **caractéristiques d'image
synthétiques**, de façon visible dans l'interface (jamais silencieuse) — le
pipeline fonctionne alors de bout en bout.

Pour activer la lecture de vraies lames `.svs`, installer en plus la
bibliothèque système OpenSlide puis `openslide-python` :

```bash
# Debian/Ubuntu
sudo apt install openslide-tools
pip install openslide-python

# macOS (Homebrew)
brew install openslide
pip install openslide-python
```

## Lancement

```bash
streamlit run app.py
```

## Données attendues

### Données omiques (CSV)

Chaque fichier a les **identifiants patientes en première colonne** (utilisée
comme index par `pandas.read_csv(..., index_col=0)`), puis une colonne par
variable :

| Fichier | Contenu | Requis |
|---|---|---|
| `data.train_mrna.csv` | expression mRNA, train | oui |
| `data.train_mirna.csv` | expression miRNA, train | oui |
| `data.train_subtype.csv` | sous-type de chaque patiente train | oui — une colonne `subtype` (ou toute colonne unique) |
| `data.test_mrna.csv` | expression mRNA, test | oui |
| `data.test_mirna.csv` | expression miRNA, test | oui |
| `data.test_subtype.csv` | sous-type de chaque patiente test | oui |
| `data.train_protein.csv` | expression protéique, train | **optionnel** — s'il y en a |
| `data.test_protein.csv` | expression protéique, test | **optionnel** — s'il y en a |

Le bloc protéines est **génériquement optionnel** de bout en bout du pipeline
(chargement, design DIABLO, `keepX`, RBC), **côté train comme côté test** :
s'il n'est fourni ni pour l'un ni pour l'autre, l'analyse tourne normalement
avec mRNA + miRNA (+ image si fournie). Dans le tutoriel officiel mixOmics
TCGA-BRCA (<https://mixomics.org/mixdiablo/diablo-tcga-case-study/>), ce bloc
n'existe qu'en train — mais si vos propres données en disposent aussi côté
test, vous pouvez le fournir via `data.test_protein.csv`.

Si aucun fichier n'est fourni (mode « Données synthétiques de démonstration »),
un jeu synthétique à 3 sous-types (Basal, Her2, LumA) est généré à la volée,
paramétrable (nombre de patientes train/test, graine aléatoire).

#### Alignement des identifiants patientes — bug corrigé

**Une version précédente contenait un bug d'alignement silencieux** : le
fichier `subtype` était chargé puis son index (les identifiants patientes)
était **écrasé** par celui du fichier mRNA dès que les deux fichiers avaient
le même NOMBRE de lignes — sans jamais vérifier qu'ils étaient dans le MÊME
ORDRE. Si votre fichier `data.train_subtype.csv` liste les patientes dans un
ordre différent de `data.train_mrna.csv` (p. ex. trié alphabétiquement par
identifiant plutôt que dans l'ordre d'export GDC), chaque étiquette se
retrouvait associée à la mauvaise patiente — **sans aucune erreur ni
avertissement**, juste un entraînement et une évaluation sur un alignement
faux. C'est le type de bug qui produit des résultats qui « ont l'air normaux »
(les chiffres restent plausibles) mais sont faux — exactement le genre
d'anomalie qui peut faire qu'une exécution sur données réelles ressemble de
façon suspecte à l'exécution sur données synthétiques par défaut.

Ce bug est corrigé : chaque bloc omique et le fichier `subtype` conservent
désormais **leur propre index** tel quel, puis sont **réalignés strictement
par identifiant patiente** (intersection des IDs présents dans tous les
fichiers requis, jamais par position ni par simple correspondance de
comptage). Tout identifiant absent d'un des fichiers est écarté et signalé
clairement (nombre écarté + quelques exemples) dans le journal d'exécution et
dans l'aperçu des données de l'interface, jamais silencieusement mal associé.

#### Séparateur de colonnes détecté automatiquement — second bug corrigé

Les fichiers CSV sont maintenant lus avec **détection automatique du
séparateur** (virgule, point-virgule, tabulation, barre verticale) : un
export « européen » (souvent produit par Excel avec `;`) est géré
correctement. Auparavant, la lecture forçait la virgule ; un fichier avec un
autre séparateur était lu comme **une seule colonne** (toute la ligne),
laquelle était ensuite consommée comme index — le bloc se retrouvait alors
avec **0 colonne de variable**, ce qui provoquait un plantage tardif et
cryptique pendant l'ajustement du modèle (`index 0 is out of bounds for axis
1 with size 0`) plutôt qu'un message clair au chargement. Si un fichier
produit malgré tout 0 colonne de variable après détection du séparateur (ou
si aucun identifiant patiente n'est commun entre les fichiers), l'application
lève maintenant une erreur explicite au moment du chargement, en nommant le
fichier en cause.

#### Fichier de sous-types sans colonne d'identifiant (troisième bug corrigé)

Certains exports de sous-types (p. ex. un simple vecteur PAM50) n'ont **aucune
colonne d'identifiant patiente** : une seule colonne de labels, dans le même
ordre que le fichier mRNA. Ce format est accepté, mais désormais **toujours
signalé explicitement** — jamais silencieusement — dans le journal
d'exécution et l'onglet « RBC — détails » : « associés PAR POSITION à
l'ordre du fichier mRNA correspondant ». Vérifiez cet avertissement si vous
en voyez un ; en cas de doute sur l'ordre des lignes, ajoutez une colonne
d'identifiant patiente au fichier de sous-types pour un alignement 100%
fiable par ID plutôt que par position.

### Données image (histopathologie, `.svs`)

Deux façons de fournir les lames :

- **Dossier local** : indiquer le chemin d'un dossier contenant les `.svs`
  train et un dossier pour les `.svs` test (pratique si l'application tourne
  sur votre propre machine, pour éviter de téléverser de gros fichiers).
- **Téléversement direct** : sélectionner plusieurs fichiers `.svs` (train et
  test séparément).

L'identifiant patiente de chaque lame est déduit du nom de fichier (partie
avant le premier `_`), exactement comme dans le script d'origine.

Si aucune lame n'est exploitable (dossier vide, openslide absent, lecture en
échec), l'application retombe sur des **caractéristiques d'image
synthétiques** — annoncé explicitement dans l'interface.

Si la case **« Ignorer les images »** est cochée, le comportement est
différent : aucun modèle omique+image n'est construit du tout (ni dense, ni
sparse, ni RBC), et aucune comparaison A/B (omique seule vs. omique+image)
n'est calculée ni affichée — seule l'analyse **omique seule** tourne
(identique à l'évaluation de référence comparable au tutoriel mixOmics). Les
onglets « Matrices de confusion — Comparaison équitable », « Comparaison
équitable » et la colonne « Config B » de l'onglet « RBC — détails »
affichent alors un simple message l'indiquant, plutôt que des résultats
vides ou trompeurs.

## Paramètres exposés

Tous les paramètres exposés dans l'interface correspondent à des paramètres
réellement réglables dans le pipeline d'origine, avec leur valeur par défaut
d'origine :

- poids de design **entre blocs** (0.1 par défaut) ;
- poids de design **vers Y** : **fixé à 1.0**, non exposé comme réglable car
  il était codé en dur dans le script d'origine ;
- nombre de composantes (`ncomp`, 2 par défaut) ;
- `keepX` (nombre de variables conservées par composante) pour le modèle
  sparse, par bloc (mRNA : [8, 5], miRNA : [10, 5], protéines : [6, 5],
  image : [8, 6]) ;
- taille de vignette pour l'extraction des caractéristiques d'image
  (256×256 par défaut) ;
- normalisation de coloration Reinhard (activée par défaut) ;
- grille de k pour le RBC (3, 5, 7, 9, 11, 15, 21 par défaut) ;
- nombre de plis / répétitions de validation croisée pour l'estimation des
  poids de bloc et la sélection de k du RBC ;
- nombre de partitions de l'étude répétée (20 par défaut).

## Résultats produits

- **Résumé** : tableau exactitude/F1 par modèle × configuration × split.
- **Matrices de confusion — Référence** : sur les 150 train / 70 test
  complètes (omique seule), **directement comparables** au tutoriel officiel
  mixOmics.
- **Matrices de confusion — Comparaison équitable** : sur le sous-ensemble de
  patientes disposant d'une image exploitable, config A (omique seule) vs.
  config B (omique + image) — **pas** directement comparables au tutoriel
  officiel (sous-ensemble de patientes différent).
- **Comparaison équitable** : graphiques en barres exactitude/F1, à
  patientes identiques.
- **Étude répétée** : `StratifiedShuffleSplit` tire 20 partitions par défaut
  (`N_REPEATS`, modifiable ; seed 0) sur toutes les 220 patientes, avec 150
  patientes d'entraînement et 70 de test à chaque partition. Les mêmes
  partitions sont réutilisées pour DIABLO dense, DIABLO sparse, RBC et les
  configurations A/B. À chaque répétition, le centrage-réduction, la
  normalisation Reinhard, DIABLO (H=2, design inter-blocs=0,1 et vers Y=1),
  ainsi que la sélection de k et des poids RBC (CV 5 plis × 2 répétitions)
  sont ajustés sans utiliser le test. Le tableau et les CSV donnent
  l'exactitude et le F1 macro (A, B et delta B−A) en entraînement et en test,
  en moyenne et écart-type d'échantillon (`ddof=1`). Chaque delta est calculé
  par partition avant agrégation. **Les protéines ne sont pas utilisées dans
  l'étude à partitions répétées, faute de disponibilité sur toutes les
  patientes.** Ces résultats ne sont donc pas directement comparables à ceux
  de la partition d'origine.
- **RBC — détails** : poids de bloc, k sélectionné, scores de validation
  croisée par k, et un **sélecteur de patiente test** pour citer ses k plus
  proches voisines (RETRIEVE du RBC) : identifiant, vraie classe, distance
  par bloc, score combiné, rang — avec le vrai sous-type, le sous-type
  prédit par le RBC et leur accord.
- **Téléchargements** : archive zip contenant tous les CSV et figures PNG,
  y compris `rbc_nearest_neighbors_omics.csv` et
  `rbc_nearest_neighbors_full.csv` (voisinage complet de chaque patiente
  test, format long, un config par fichier).

De plus, dès qu'une exécution démarre, un **aperçu des données chargées**
s'affiche (nombre de patientes par bloc train/test, blocs réellement en jeu,
premiers identifiants de chaque bloc, distribution des classes train/test) —
visible pendant l'exécution et conservé dans l'onglet Résumé après coup. Cela
permet de vérifier immédiatement que ce sont bien vos données (et pas le jeu
synthétique) qui ont été utilisées, et que le bloc protéines est bien présent
si vous l'avez fourni.

## Citer les patientes les plus proches (RBC)

`CaseBase.retrieve_neighbors` (et la fonction `retrieve_neighbors_batch`)
exposent le RETRIEVE du RBC : pour une patiente donnée, les k plus proches
voisines de la base de cas, avec leur identifiant, leur vraie classe, la
distance par bloc et le score combiné pondéré. Ces fonctions réutilisent
**exactement** le même calcul de similarité (`CaseBase.similarity`) que
`diagnose`/`diagnose_batch`, qui produisent la prédiction RBC affichée par
ailleurs — les voisins cités sont donc garantis être ceux qui pilotent
réellement la décision. Dans `run_pipeline`, ce voisinage est précalculé pour
**toutes les patientes test**, pour les deux configurations (omique seule et
omique + image, chacune avec sa propre base de cas / ses propres poids / son
propre k), et stocké dans `PipelineResults.neighbors_test`.

## Avertissement méthodologique important (RBC)

Le RBC ne retrouve pas ses cas voisins dans les variables omiques brutes,
mais dans les **scores latents à 2 dimensions par bloc** appris par
`block.plsda` (dense) — exactement l'espace que DIABLO utilise lui-même pour
classer via ses centroïdes. C'est un k-NN pondéré par bloc, appliqué à une
représentation déjà optimisée pour séparer les sous-types, au lieu des
données brutes.

**Implication méthodologique à mentionner dans l'article** : le RBC n'est
alors plus un algorithme totalement indépendant de DIABLO, c'est un
raisonnement à base de cas appliqué à la représentation apprise par DIABLO
(une forme d'« apprentissage de métrique » supervisé avant k-NN, technique
courante et légitime). Le RBC garde son cycle Retrieve/Reuse et reste
interprétable (on peut toujours citer les patientes les plus proches), mais
la comparaison « RBC vs DIABLO » doit être présentée comme une comparaison
entre deux stratégies de **décision** sur la même représentation apprise
(centroïde de classe vs. vote pondéré des k plus proches voisins), pas comme
deux méthodes indépendantes « from scratch ».

## Fichiers

- `diablo_core.py` — logique réutilisable (chargement de données, pipeline
  image, `BlockPLSDA`, `CaseBase`, métriques, figures), sans effet de bord.
- `app.py` — interface Streamlit.
- `requirements.txt` — dépendances.
# mutilomics-article
# DIABLO-et-RBC-pour-l-int-gration-multi-omique-explicable
# DIABLO-et-RBC-pour-l-int-gration-multi-omique-explicable
# DIABLO-et-RBC-pour-l-int-gration-multi-omique-explicable
