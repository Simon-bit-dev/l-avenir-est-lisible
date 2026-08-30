
# Arbre existentiel probabiliste — V2

## Lancement

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 -m streamlit run app.py
```

## Changements principaux

- Les 6 561 feuilles restent calculées en arrière-plan.
- La sortie principale est désormais narrative.
- Détection de tensions internes.
- Trois scénarios à 12 mois.
- Bifurcations les plus probables.
- Leviers contextuels.
- Matrice de sensibilité différente selon chaque dimension.
- Comparaison entre intuition subjective et projection du modèle.

## Limite
La projection à 12 mois est encore heuristique et ne doit pas être interprétée comme une probabilité scientifiquement calibrée.
