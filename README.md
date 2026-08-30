# Arbre existentiel probabiliste — V3

## Nouveauté principale

La restitution situe désormais visuellement le répondant dans la structure ternaire :
- trois branches visibles à chaque niveau ;
- chemin dominant accentué ;
- probabilités locales visibles au survol ;
- coordonnée ontologique du type `1.3.1.2.1.1.2.1`.

L'arbre complet à 8 dimensions contient 9 841 nœuds, dont 6 561 feuilles.
L'interface affiche une projection locale de l'arbre afin de rester lisible.

## Lancement

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 -m streamlit run app.py
```

## Déploiement Streamlit Cloud

Placez `app.py`, `requirements.txt` et `README.md` à la racine du dépôt GitHub,
puis sélectionnez `app.py` comme fichier principal dans Streamlit Community Cloud.

## Limite

Les probabilités futures restent heuristiques et ne sont pas calibrées sur des données longitudinales.
