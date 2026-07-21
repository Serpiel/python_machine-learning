import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# --- 1. ---
df = pd.read_csv('ventes.csv')

# --- 2. ---
print("=== ANALYSE EXPLORATOIRE ===")
# Dimensions
print(f"Dimensions du jeu de données : {df.shape[0]} lignes et {df.shape[1]} colonnes.")

# Valeurs manquantes
print("\nRecherche des valeurs manquantes :")
print(df.isnull().sum())

# Statistiques descriptives
print("\nStatistiques descriptives :")
print(df.describe(), "\n")

# --- 3. ---
colonnes_numeriques = ['temperature', 'budget_publicitaire', 'ventes']
matrice_corr = df[colonnes_numeriques].corr()

print("=== MATRICE DE CORRÉLATION ===")
print(matrice_corr, "\n")

# --- 4. ---
plt.figure(figsize=(7, 5))
# matshow affiche une matrice sous forme d'image (carte de chaleur)
cax = plt.matshow(matrice_corr, cmap='coolwarm', fignum=1)
plt.colorbar(cax, label='Coefficient de corrélation')

# Ajout des labels sur les axes pour la lisibilité
plt.xticks(range(len(colonnes_numeriques)), colonnes_numeriques, rotation=15)
plt.yticks(range(len(colonnes_numeriques)), colonnes_numeriques)
plt.title('Carte de chaleur des corrélations', pad=20)
plt.show()

# --- 5. ---
# Variables explicatives (X) et variable cible (y)
X = df[['temperature', 'budget_publicitaire']]
y = df['ventes']

# Séparation des données (80% entraînement, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Création et entraînement du modèle
modele = LinearRegression()
modele.fit(X_train, y_train)

# --- 6 & 7. ---
y_pred = modele.predict(X_test)
score_r2 = r2_score(y_test, y_pred)

# Récupération des coefficients
coef_temperature = modele.coef_[0]
coef_budget = modele.coef_[1]

print("=== RÉSULTATS DE LA RÉGRESSION ===")
print(f"Score R² : {score_r2:.4f}")
print(f"Coefficient Température : {coef_temperature:.4f}")
print(f"Coefficient Budget Pub  : {coef_budget:.4f}\n")

# --- 8. ---
print("=== INTERPRÉTATION SIMPLE ===")
print(f"• Fiabilité : Le modèle explique {score_r2 * 100:.1f}% de la variation des ventes (Score R²).")
print(f"• Impact Température : À budget constant, une hausse de 1 degré modifie les ventes de {coef_temperature:.2f} unités.")
print(f"• Impact Budget : À température constante, un investissement de 1 unité supplémentaire dans le budget publicitaire modifie les ventes de {coef_budget:.2f} unités.")