# Projet series chronologiques

Livrables principaux :

- `rapport/rapport_ventes_voitures.tex` : rapport LaTeX pret a compiler.
- `notebooks/box_jenkins_ventes_voitures_python.ipynb` : notebook Python.
- `notebooks/box_jenkins_ventes_voitures_R.Rmd` : notebook R.
- `data/ventes_voitures_quebec.txt` : serie utilisee au format texte.
- `figures/` : figures inserees dans le rapport.
- `outputs/` : tableaux CSV des tests, comparaisons de modeles et previsions.
- `scripts/analyse_ventes_voitures.py` : script Python pour regenerer les resultats.

Commande Python pour regenerer les figures et tableaux :

```powershell
.\.venv\Scripts\python.exe .\scripts\analyse_ventes_voitures.py
```

Compilation LaTeX :

```powershell
cd rapport
pdflatex rapport_ventes_voitures.tex
pdflatex rapport_ventes_voitures.tex
```

Remarque : `pdflatex` n'est pas installe dans l'environnement actuel. Le fichier
`.tex` peut etre compile sur Overleaf ou avec une distribution LaTeX locale.
