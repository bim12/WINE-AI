# 🍇 Wine-AI

Projet d’analyse, de prédiction et de classification pour la viticulture, combinant **Machine Learning**, **Deep Learning** et une API web avec **FastAPI**

## 📂 Structure du projet
<pre>
Wine-AI/
├── app.py # API FastAPI (serveur)
├── debug.ipynb # Notebook d’analyse/soutenance
├── DL/ # Deep Learning: scripts, notebooks, datasets (images)
│ └── Grapevine Disease Dataset Original Data/
├── ML/ # Machine Learning: scripts, notebooks
├── Models/ # Modèles sauvegardés (.h5, .pt, .pkl, etc.)
├── Images_web/ # Images diverses (visualisations, outputs)
├── requirements.txt # Dépendances Python du projet
└── ... # (autres fichiers, .gitignore, etc.)
</pre>
## 🚀 Lancer l’API (FastAPI)

**1. Installer les dépendances**
```bash
python -m venv .venv
source .venv/Scripts/activate   # Sous Windows
pip install -r requirements.txt

2. Lancer le serveur FastAPI
L’API sera disponible sur : http://localhost:8888/docs
Lancer l’API FastAPI dans l’environnement virtuel
python app.py
ou
uvicorn app:app --reload
```


## 🤖 Fonctionnalités
Machine Learning & Deep Learning :

Prédiction et classification de maladies de la vigne

Utilisation de réseaux de neurones, SVM, RandomForest, etc.

Traitement d’images (dataset : Grapevine Disease)

API Web (FastAPI) :

Endpoints pour prédiction, upload, analyse de résultats

Format REST, documentation automatique Swagger

Notebooks interactifs :

Pour analyse, exploration de données, démonstration (soutenance)

Gestion des modèles :

Chargement/sauvegarde des modèles entraînés (fichiers dans /Models/)

## 💾 Données & Images
Les images du dataset sont incluses dans DL/Grapevine Disease Dataset Original Data/

Autres images (outputs, figures) : Images_web/

Aucun jeu de données privé ou sensible n’est inclus dans le repo public

## 🧑‍💻 Utilisation typique
Entraîner les modèles via les notebooks ou scripts dans /ML et /DL

Sauvegarder les modèles dans /Models

Lancer l’API avec app.py (FastAPI)

Tester les endpoints (prévoir un client HTTP ou via /docs)

### 🛠️ Dépendances principales
Python ≥ 3.10

numpy, pandas, scikit-learn, tensorflow/keras, torch, matplotlib, opencv-python, fastapi, uvicorn, etc.

(voir requirements.txt)

## 🙌 Contributeurs
Auteur principal : [bim12]

📝 Notes
Le projet est prêt à être déployé/présenté en l’état

Notebooks à jour pour la démo

API testable localement avec les datasets inclus

## 📄 License
Projet open-source
