# 📊 MetroSignal — Analyse Exploratoire des Données (EDA)

**Dataset** :  64 253 252 lignes · 779 stations · 2015–2025  
**Pipeline** : Toutes les agrégations sont réalisées directement en SQL via DuckDB. Pandas ne reçoit que les résultats finaux pour éviter le goulot mémoire (~16 GB si chargement complet).

---

## 1. Top 20 Stations — Volume Total de Validations

![Top 20](/outputs\img\eda\2_1_top20_stations.png)

**Saint-Lazare domine largement** avec ~780M de validations sur 11 ans, presque le double de La Défense-Grande Arche (~544M). Les grandes gares terminus (Lyon, Nord, Est, Montparnasse) et les pôles d'échange majeurs (Châtelet-Les Halles) occupent les rangs suivants.

**Limite connue — La Défense surestimée** : La Défense-Grande Arche agrège plusieurs codes arrêts correspondant à des lignes distinctes (RER A code 393/394, Métro 1 code 414, Transilien). Son volume (~544M) est artificiellement élevé par rapport aux stations mono-ligne. À prendre en compte lors de l'interprétation du classement par volume.

**Implication ML** : entraîner des modèles par station ou utiliser `rang_station` comme feature pour éviter que les grandes stations dominent les métriques globales.

---

## 2. Validations Moyennes par Heure — Semaine vs Weekend

![Validations Moyennes](/outputs\img\eda\2_2_heures_pointe.png)

Le profil semaine présente le **double pic classique** à 8h (pic à ~840 validations/station) et 17h–18h (~760), avec un creux marqué à 9h–10h après le rush matinal. Le profil weekend est radicalement différent : pas de pic matinal, montée progressive jusqu'à un plateau entre 14h et 19h (~340 validations/station).

**Anomalie axe X** : l'axe affiche -1 et 24 en dehors de la plage réelle 0–23. Artefact d'affichage Plotly corrigé avec `range=[-0.5, 23.5]` sur les figures suivantes.

**Implication ML** : `heure` et `is_weekend` seront parmi les features les plus importantes. L'encodage cyclique sin/cos est obligatoire pour que le modèle comprenne que 23h et 0h sont adjacentes.

**Note sur les créneaux 2h–5h** : le réseau RATP n'opère pas sur ces créneaux. Les ~7.3M de lignes correspondantes ont `taux_congestion = NaN` après invalidation des créneaux avec `baseline_mean < 1 validation` (division par quasi-zéro produisant des z-scores aberrants jusqu'à +23).

---

## 3. Heatmap Taux de Congestion — Top 30 Stations × Heure

![Heatmap](/outputs\img\eda\2_3_heatmap_station_heure.png)

La heatmap révèle une structure globalement homogène (teinte jaune/neutre = z-score ≈ 0) avec quelques anomalies localisées :

- **LA DEFENSE-GRANDE ARCHE et ESPLANADE DE LA DEFENSE** : taches noires à 2h–4h indiquant des z-scores très négatifs. Cause probable : la reconstruction horaire via `PROFIL_FER` attribue un pourcentage infime (~0.003%) aux créneaux nocturnes. Sur certaines stations, quelques validations isolées à ces heures (nettoyeurs, agents RATP) produisent des z-scores extrêmes malgré le fix `baseline_mean < 1`. À investiguer en Phase 4.
- **BELLEVILLE** : tache rouge à 3h–4h (z-score positif). Même cause — une poignée de validations nocturnes sur une baseline quasi-nulle.

La majorité des stations présentent un profil stable, confirmant que le z-score est bien calibré sur la période 2015–2025.

---

## 4. Distribution du Taux de Congestion par Jour de la Semaine

![Série Temporelle](/outputs/img/eda/2_4_distribution_congestion.png)

Les jours ouvrés (Lundi–Vendredi) présentent une distribution **resserrée et symétrique** autour de 0, avec des moustaches atteignant ±2.5 z-scores. Le weekend (Samedi–Dimanche) se distingue par :

- Une **médiane légèrement négative** (~-0.1) : le trafic weekend est structurellement sous la normale des jours ouvrés
- Une **variance nettement plus large** : les moustaches descendent jusqu'à -4, signe d'une plus grande imprévisibilité

**Note technique** : la version initiale de cette figure utilisait `go.Box` avec des percentiles précalculés en SQL, ce qui causait la superposition de toutes les boîtes sur x=0 (absence de coordonnée x explicite dans Plotly). Remplacé par `px.box` sur un échantillon de 200 000 lignes tiré avec `USING SAMPLE`.

---

## 5. Série Temporelle Globale — Validations Journalières 2015–2025

![serie temporelle](/outputs/img/eda/3_1_serie_temporelle.png)

Quatre événements historiques sont clairement lisibles :

**COVID-19 (mars 2020)** : effondrement brutal à ~200K validations/jour (vs ~5–6M normalement), soit une chute de 96%. La récupération est progressive et n'atteint pas le niveau pré-COVID avant fin 2022, traduisant un changement durable des habitudes (télétravail, vélo).

**Grève RATP décembre 2019 – janvier 2020** : dense cluster de marqueurs rouges sur ~40 jours. La grève des transports contre la réforme des retraites est la plus longue de l'histoire de la RATP.

**JO Paris 2024 (juillet–août)** : pic visible à ~8–10M validations/jour, nettement au-dessus de la normale estivale.

**Spike post-2023** : hausse structurelle du trafic à partir de 2023 (~7–8M vs ~5–6M avant COVID). Probablement lié à l'extension du périmètre IDFM (intégration de nouvelles lignes) plutôt qu'à une croissance organique — à documenter dans les limites du projet.

**Note sur la détection de grèves** : les marqueurs rouges pendant le confinement COVID 2020 étaient initialement des faux positifs (trafic quasi-nul détecté comme grève). Corrigé par l'ajout du flag `is_covid` et l'exclusion des périodes COVID dans la fonction `detect_greve()`.

---

## 6. Nombre de Jours de Grève Détectés par Année

![grève](/outputs/img/eda/3_2_greves_par_an.png)

Après correction du flag `is_covid` :

- **2019 : 30 jours** — cohérent avec la grève RATP de décembre
- **2020 : 60 jours** — réduit depuis 124 (avant correction COVID). Les 60 restants correspondent aux grèves sociales réelles hors confinements
- **2021 : 51 jours** — légèrement élevé, possiblement des perturbations liées au contexte post-COVID
- **2022–2025 : 5–10 jours/an** — retour à un niveau normal

**Seuil de détection** : un jour est flaggé `is_greve = 1` si ≥50% des stations actives ont `taux_congestion < -1.55`.

---

## 7. Indice de Fréquentation Mensuel (base 100 = 2019)

![JO Paris 2024](/outputs/img/eda/3_3_indice_frequentation.png)

**2020 en jaune** : chute à 5–10 en avril (confinement strict), remontée progressive mais le niveau 2019 n'est jamais retrouvé sur l'année.

**Pic de décembre pour toutes les années** : artefact de la baseline. Décembre 2019 est le pire mois de la grève RATP — son total de validations est anormalement bas, ce qui déprime la base 100 de ce mois. Toutes les autres années ayant un décembre normal, leur indice explose mécaniquement. **Ce n'est pas un bug à corriger** : c'est une information réelle sur l'impact de la grève, documentée par une annotation sur le graphe.

**Croissance 2015–2019** : les années antérieures sont légèrement sous 100 (80–90), confirmant une croissance organique du trafic sur la période.

---

## 8. JO Paris 2024 — Stations avec la Plus Forte Hausse vs Été 2023

![JO Paris 2024](/outputs/img/eda/3_4_jo_2024.png)

Les stations les plus impactées sont cohérentes avec la localisation des sites olympiques :

- **LE STADE** et **DOURDAN-LA-FORET** : delta > 4 z-scores — accès direct aux sites de compétition
- **PORTE DE PANTIN** et **PORTE DE LA CHAPELLE** : accès au Stade de France et aux sites nord-parisiens
- **PORTE D'AUTEUIL** : Roland-Garros (tennis)

L'impact est réel et mesurable sur 16 jours (26 juillet – 11 août 2024), ce qui **valide l'utilité de la feature `nb_events`** pour le modèle ML. Les événements de grande ampleur laissent une signature claire dans les données de validation.

---

## 9. Insight Clé : Météo vs Heure — Corrélation avec le Taux de Congestion

![Insight Clé](/outputs/img/eda/4_1_meteo_vs_heure.png)

Toutes les stations apparaissent dans le quadrant "Météo > Heure" (axe X ≈ 0, axe Y > 0). Ce résultat à ~100% est un **artefact méthodologique**, pas un résultat substantiel.

**Explication** : le `taux_congestion` est un z-score calculé par créneau `(station, jour_semaine, heure)`. Par construction, la moyenne de `taux_congestion` par heure est mécaniquement 0 pour chaque heure (vérification : toutes les heures ont `AVG(taux_congestion) ≈ -0.001`). La corrélation de Pearson entre `heure` et une variable dont la moyenne conditionnelle est constante est nécessairement quasi-nulle.

**Pearson est le mauvais outil ici.** Pour la Phase 4, recalculer avec :
- Encodage cyclique `heure_sin / heure_cos` (déjà présents dans le dataset)
- Corrélation de Spearman pour les features météo
- Mutual Information comme alternative non-paramétrique

L'hypothèse "certaines stations aériennes sont plus sensibles à la météo qu'à l'heure" reste valide — elle sera testée correctement en Phase 4.

---

## 10. Intensité des Précipitations vs Taux de Congestion

![Intensité précipitations](/outputs/img/eda/4_2_pluie_intensite.png)

L'effet des précipitations sur le taux de congestion est **statistiquement faible à l'échelle globale** — les distributions "Sec", "Légère", "Modérée", "Forte" se superposent largement. La catégorie "Très forte" présente une variance réduite, probablement due à un faible nombre d'observations.

**Note technique** : même bug que le boxplot jour de la semaine — `go.Box` avec percentiles précalculés superpose tout sur x=0. Remplacé par `px.box` sur échantillon 200K lignes.

Le signal météo sera probablement plus fort au niveau station qu'au niveau réseau global, notamment pour les stations aériennes (ligne 6, certaines stations RER).

---

## 11. Impact de la Pluie par Heure — Δ Taux Congestion (Pluie − Sec)

![Impact pluie par heure](/outputs/img/eda/4_3_delta_pluie_heure.png)

La pluie a un **effet positif cohérent sur le trafic** (+0.01 à +0.07 z-score) : les gens prennent davantage le métro sous la pluie. L'effet est le plus fort entre 13h et 20h (+0.04 à +0.07).

**Effet négatif à 8h** (delta ≈ -0.04) : contre-intuitif mais explicable — le réseau est déjà saturé à l'heure de pointe les jours secs, limitant l'effet marginal de la pluie. À 8h sous la pluie, certains usagers habituels décalent leur départ ou cherchent d'autres modes.

**Créneaux 3h–5h** : deltas négatifs parasites dus aux `taux_congestion = NaN` (créneaux nocturnes invalides) qui faussent la moyenne conditionnelle. À neutraliser en Phase 4 avec un filtre `WHERE taux_congestion IS NOT NULL`.

---

## 12. Top 30 Stations les Plus Imprévisibles (Variance du z-score)

![Top 30 stations](/outputs/img/eda/5_1_stations_imprev.png)

**ROSNY-BOIS-PERRIER** en tête avec variance ≈ 1.0, suivi de stations de grande couronne (THIEUX-NANTOUILLET, BLANC-MESNIL, VOSGES, CDG 2-TGV). Ce résultat est **contre-intuitif** : on attendait les grandes gares parisiennes en tête.

**Explication** : la variance est calculée sur le `taux_congestion` (z-score), pas sur le volume brut. Les stations de grande couronne ont des profils très variables — elles sont très fréquentées certains jours (événements, matchs au Stade de France pour les stations proches) et quasi-vides d'autres jours. Les grandes gares parisiennes ont un flux plus régulier et prévisible malgré leur volume.

**Note** : après correction du bug `variance_historique` (calcul initial sur `nb_vald_heure` brut au lieu de `taux_congestion`), les valeurs sont maintenant entre 0.43 et 1.0, directement comparables entre stations.

---

## 13. Volume vs Variance — Stations Stratégiques pour le ML

![Volume vs Variance](/outputs/img/eda/5_2_volume_vs_variance.png)

Le scatter révèle deux populations :

- **Cluster dense en haut à gauche** : la majorité des stations (~750) avec un faible volume et une variance élevée (~0.95–1.0)
- **Outlier isolé en haut à droite** : LA DEFENSE-GRANDE ARCHE (~8000 validations/heure, variance ~0.95) — grande station avec taux de grève élevé (rouge foncé)

La corrélation volume/variance est **faiblement positive** contrairement à l'hypothèse initiale. Les stations stratégiques pour le ML sont celles à fort volume (grandes gares) car elles concentrent le trafic opérationnel, même si elles ne sont pas les plus imprévisibles en termes de z-score.

**Les médianes** (lignes pointillées) divisent l'espace en 4 quadrants. Le quadrant haut-droite (fort volume + forte variance) est vide — aucune grande gare n'est vraiment imprévisible sur l'échelle z-score.

---

## 14. Profil Horaire : Station Imprévisible vs Station Stable (±1σ)

![profil horaire station horaire](/outputs/img/eda/5_3_profil_horaire_imprev.png)

**ROSNY-BOIS-PERRIER** (imprévisible) : double pic classique 8h/17h avec une bande ±1σ très large (~±300 validations). Certains jours quasi-vides (grèves, COVID), d'autres très chargés (JO, événements) — la bande traduit cette variabilité extrême.

**PORTE D'AUTEUIL** (stable) : profil plat à ~50–130 validations/heure avec une bande ±1σ quasi-invisible. Station de quartier résidentiel à flux très régulier.

**Anomalie axe X** : -1 et 24 visibles — même artefact Plotly que figure 2, à corriger avec `range=[-0.5, 23.5]`.

---

## 15. Corrélation Pearson des Features avec taux_congestion

![Pearson](/outputs/img/eda/6_1_feature_correlations.png)

**Features à signal fort** (en valeur absolue) :
- `is_greve` : -0.35 — la grève réduit massivement le trafic
- `is_vacances_scolaires` : -0.22 — les vacances réduisent significativement le trafic
- `is_jour_ferie` : -0.20 — même effet
- `mois_cos` : +0.11 — saisonnalité annuelle (hiver = pic, été = creux)
- `temp` : +0.10 — les journées chaudes réduisent légèrement le trafic (vélo, marche)

**Features à signal quasi-nul** : toutes les features temporelles (`heure`, `heure_sin`, `heure_cos`, `jour_semaine`, `jour_sin`, `jour_cos`). Ceci n'est **pas un bug** — c'est une conséquence directe de la normalisation : le z-score est calculé par `(station, jour_semaine, heure)`, donc la moyenne conditionnelle de `taux_congestion` par heure est mécaniquement 0. Pearson ne peut pas détecter un signal sur une variable utilisée dans la normalisation de la cible.

**Conclusion** : Pearson n'est pas adapté pour évaluer l'importance des features temporelles sur ce dataset. LightGBM capturera les interactions et non-linéarités automatiquement via les splits sur les arbres. Les SHAP values confirmeront l'importance réelle de chaque feature en Phase 4.
