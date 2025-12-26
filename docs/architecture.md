Architecture (vide)
===================

But
---
Definir une base modulaire pour un pipeline RSS qui regroupe des articles en
clusters thematiques, extrait une idee par cluster et conserve l'historique.

Principes clefs
---------------
- Composition: chaque etape est un module isole et remplacable.
- Legerete: modeles simples, peu gourmands, execution locale.
- Fenetre glissante + historique complet: on analyse en "actif" les articles
  recents tout en conservant la base totale.
- Robustesse: RSS imparfaits, contenu heterogene, pas de scraping.

Modules
-------
Ingestion
- rss_reader: lit les flux RSS et normalise les champs.
- html_cleaner: nettoyage HTML et limitation de taille.

NLP
- lang_detect: detection de langue (fastText, titre uniquement).
- summarize: resume extractif sur une portion de texte.

Features
- vectorize: TF-IDF sur titres + resumes.

Clustering
- clusterer: clustering a seuil variable (distance).
- cluster_store: index/centroid par cluster, persistants.

Idees
- idea_extractor: idee par cluster a partir des resumes.

Scoring
- score: score en fonction de la taille et de la recence.

Vues
- renderer: affichage des clusters et des recents.

Storage
- models: schemas simples (article, cluster, univers).
- repository: stockage append-only (historique complet).

Utils
- time_window: gestion des fenetres glissantes.

Flux de donnees
--------------
RSS -> nettoyage -> detection langue -> resume -> TF-IDF -> clustering
   -> idee cluster -> score -> stockage -> vues

Fenetre glissante
-----------------
- L'historique complet est conserve (append-only).
- Une fenetre glissante (ex: 7/30 jours) alimente le clustering "actif".
- Les clusters historiques restent consultables.

Univers
-------
Un univers est defini par un fichier texte contenant des URLs de flux RSS.
Chaque univers peut etre traite independamment pour eviter les melanges.
