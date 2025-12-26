INFO_RECAP
==========

Objectif
--------
Projet d'aggregation RSS pour produire des univers d'information resumes et
regroupes par thematique. L'architecture privilegie la composition de modules
simples, adaptes a de petites configurations.

Principes
---------
- Univers: un univers est defini par une liste de flux RSS (fichier texte).
- Ingestion sans scraping: on recupere ce que fournit le RSS, puis on nettoie.
- NLP leger: detection de langue, resume extractif, vectorisation simple.
- Clustering variable: le nombre de clusters depend des similarites.
- Historique complet: on conserve tous les clusters, avec une fenetre glissante
  pour l'analyse "active" des articles recents.

Structure (vide)
----------------
docs/architecture.md          Description de l'architecture et des modules
config/universes/             Listes de flux RSS par univers
data/                         Donnees brutes (placeholder)
output/                       Sorties (clusters, vues) (placeholder)
src/                          Code source modulaire (placeholders)

Flux logique (future implementation)
-----------------------------------
1) Ingestion RSS -> normalisation -> nettoyage HTML
2) Detection langue + resume extractif
3) Vectorisation TF-IDF
4) Clustering par distance (seuil)
5) Extraction d'idee de cluster (keywords + resume global)
6) Vues: clusters, recents, tendances

Demarrage rapide
----------------
- Definir un univers via `config/universes/*.txt`
- Lancer l'ingest: `python -m src.cli ingest --universe <nom>`
- Detection langue (fastText sur le titre):
  `python -m src.cli ingest --universe <nom> --detect-lang --fasttext-model <path>`

Notes
-----
- Les dependances viseront des bibliotheques legeres (pandas, scikit-learn,
  sumy, fastText ou alternatives).
- Pas de web scraping: uniquement les donnees du RSS + nettoyage HTML.
