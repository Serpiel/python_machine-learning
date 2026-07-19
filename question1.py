import numpy as np

# Ventes journalières
ventes_liste = [12, 15, 14, 10, 18, 20, 17, 16, 13, 19, 22, 21, 18, 17, 23]

# --- 1. ---
ventes = np.array(ventes_liste) # convertion liste -> tableau NumPy

# --- 2. ---
moyenne = np.mean(ventes)
mediane = np.median(ventes)
ecart_type = np.std(ventes) 

# --- 3. ---
ventes_sup_moyenne = ventes[ventes > moyenne] # booléen permettant de filtre si ventes > moyenne

# --- 4. ---
ventes_normalisees = (ventes - moyenne) / ecart_type # la formule (A - A bar) / sigma A 


# --- 5. ---

print(f"Tableau initial NumPy : \n{ventes}\n")

print("--- Calculs des ventes ---")
print(f"Moyenne    : {moyenne:.2f}")
print(f"Médiane    : {mediane:.2f}")
print(f"Écart-type : {ecart_type:.2f}\n")

print("--- Ventes supérieures à la moyenne ---")
print(f"Valeurs filtrées : {ventes_sup_moyenne}\n")

print("--- Données normalisées ---")
print(np.round(ventes_normalisees, 3))