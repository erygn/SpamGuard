# SpamGuard

SpamGuard est un projet de classification de messages qui identifie si un message est du spam, du ham (message normal) ou important. Ce projet utilise un modèle de machine learning basé sur l'algorithme Naive Bayes pour la classification des textes.

## Installation

### Prérequis

- Python 3.7+
- pip (gestionnaire de paquets Python)
- Docker (pour exécuter l'application avec Nginx)

### Étapes d'installation

1. Clonez le dépôt :

```
git clone https://github.com/erygn/SpamGuard.git
cd SpamGuard
```

2. Installez les dépendances :
```
python3 -m pip install -r requirements.txt
```

3. Préparez les données et entraînez le modèle :
```
python model/train.py
```

4. Lancez l'application Flask :
```
python app.py
```

L'application sera disponible sur http://127.0.0.1:5000.

## Utilisation
Endpoint de prédiction, pour prédire la catégorie d'un message, utilisez l'endpoint /predict.

### Requête

- Méthode : POST
- URL : http://127.0.0.1:5000/predict
- Corps de la requête (JSON) :
```
{
  "message": "Votre message ici"
}
```

### Réponse

- Corps de la réponse (JSON) :
```
{
  "prediction": "spam/ham/important",
  "contributions": [
    ["mot1", contribution1],
    ["mot2", contribution2],
    ...
  ]
}
```