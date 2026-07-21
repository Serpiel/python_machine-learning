import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# --- 1. ---
df = pd.read_csv('clients.csv')

# --- 2. ---
X = df[['age']] 
y = df['montant_achat']

# --- 3. ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 4. ---
modele = LinearRegression()
modele.fit(X_train, y_train)

# --- 5. ---
y_pred = modele.predict(X_test)

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("=== ÉVALUATION DU MODÈLE DE RÉGRESSION ===")
print(f"Coefficient de détermination (R²) : {r2:.4f}")
print(f"Erreur quadratique moyenne (MSE)  : {mse:.2f}")

# --- 6. ---
plt.figure(figsize=(8, 5))

plt.scatter(X, y, color='blue', label='Données réelles', alpha=0.7)

plt.plot(X, modele.predict(X), color='red', linewidth=2, label='Droite de régression')

plt.title('Régression Linéaire : Prédiction du montant d\'achat selon l\'âge')
plt.xlabel('Âge')
plt.ylabel('Montant d\'achat (€)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

plt.show()