import pandas as pd

# --- 1. ---
df = pd.read_csv('clients.csv')

# --- 2. ---
print("--- Informations générales ---")

nb_clients = len(df)
print(f"Nombre de clients : {nb_clients}\n")

print("--- 5 premières lignes ---")
print(df.head(5), "\n")

print ("--- Statistiques descriptives ---")
print(df.describe(), "\n")

# --- 3. ---
print("--- Montant moyen par ville ---")

moyenne_ville = df.groupby('ville')['montant_achat'].mean()
print(moyenne_ville, "\n")

# --- 4. ---
clients_premium = df[df['montant_achat'] > 500]

# --- 5. ---
clients_premium.to_csv('clients_premium.csv', index=False)
print("Le fichier 'clients_premium.csv' a été créé avec succès.")