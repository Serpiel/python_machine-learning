import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('clients.csv')

# --- 1. ---
plt.figure(figsize=(8, 5))
plt.hist (df['age'], bins=5, color='skyblue', edgecolor='black')

# Graphique
plt.title("Distribution des âges des clients")
plt.xlabel("Âge")
plt.ylabel("Nombre de clients")
plt.grid(axis='y', alpha=0.7) # affichage à l'horizontale pour une meilleure lisibilité
plt.show()

# --- 2. ---
plt.figure(figsize=(8, 5))
plt.scatter(df['age'], df['montant_achat'], color='orange', alpha=0.8)

# Graphique
plt.title("Montant d'achat en fonction de l'âge")
plt.xlabel('Âge')
plt.ylabel("Montant d'achat (€)")
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# --- 3. ---
moyenne_par_ville = df.groupby('ville')['montant_achat'].mean()
plt.figure(figsize=(8, 5))
plt.bar(moyenne_par_ville.index, moyenne_par_ville.values, color='lightgreen', edgecolor='black')

# Graphique
plt.title('Montant moyen des achats par ville')
plt.xlabel('Ville')
plt.ylabel('Montant moyen (€)')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()