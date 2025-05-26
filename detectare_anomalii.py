"""
Proiect: D7-T2 | Detecția anomaliilor
Echipa: 21-E6
Student: DEJICA P. ANDREI-RADU

Sursa date: https://www.kaggle.com/datasets/shashwatwork/cerebral-stroke-predictionimbalaced-dataset
Sursa IsolationForest: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html#sklearn.ensemble.IsolationForest
Sursa IsolationForest2:https://medium.com/@corymaklin/isolation-forest-799fceacdda4
Sursa z-score: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.zscore.html
Sursa grafice: https://matplotlib.org/stable/gallery/lines_bars_and_markers/bar_label_demo.htmlklearn.ensemble.IsolationForest
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib
import matplotlib.pyplot as plt

#Primul pas pe care il facem, folosind pandas extragem fisierul CSV pentru a il citi.
#In cazul in care se numeste diferit de 'dataset.csv' schimbam numele fisierului dinauntrul parantezelor din linia 17
data = pd.read_csv('dataset.csv')
print('Datele au fost incarcate. Shape:', data.shape)

from scipy.stats import zscore

# Selecteaza doar coloanele numerice
numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
print('Coloane numerice:', numeric_cols)

# Calculeaza z-score pentru fiecare coloana numerica
z_scores = np.abs(zscore(data[numeric_cols], nan_policy='omit'))
outliers_zscore = (z_scores > 3) # Considerăm outlier orice valoare cu z-score > 3
print('Numar de outliers (z-score > 3) pe coloana:')
for idx, col in enumerate(numeric_cols):
    print(f"  {col}: {outliers_zscore[:, idx].sum()}")

# Facem un grafic aici pentru outliers, sa vedem mai bine numerele
outliers_count = [outliers_zscore[:, idx].sum() for idx in range(len(numeric_cols))]
plt.figure(figsize=(10, 5))
plt.bar(numeric_cols, outliers_count, color='#eebbc3')
plt.title('Numar de outliers (z-score > 3) pe coloana')
plt.ylabel('Numar outliers')
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig('outliers_zscore_pe_coloana.png')
plt.close()

# Aplicam isolation Forest pentru deteclarea de outliers
# Folosim doar coloanele numerice pentru model
iso_forest = IsolationForest(contamination='auto', random_state=42)
iso_labels = iso_forest.fit_predict(data[numeric_cols].fillna(0))  # -1 = outlier, 1 = normal

data['is_outlier_iso'] = iso_labels
print('\nDistributie outliers dupa Isolation Forest:')
print(data['is_outlier_iso'].value_counts())

# Grafic pentru distribuaia outliers dupa Isolation Forest
plt.figure(figsize=(5, 4))
data['is_outlier_iso'].replace({-1: 'Outlier', 1: 'Normal'}).value_counts().plot(kind='bar', color=['#eebbc3', '#232946'])
plt.title('Distributie outlieri Isolation Forest')
plt.ylabel('Numar')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('outliers_isoforest.png')
plt.close()

# Dacă exista o coloana de clasa minoritara (ex: 'stroke'), putem compara cu outlierii
if 'stroke' in data.columns:
    print('\nAnaliza outliers vs. clasa minoritara (stroke):')
    print(pd.crosstab(data['stroke'], data['is_outlier_iso'], rownames=['stroke'], colnames=['IsolationForest']))
else:
    print("\nColoana 'stroke' nu a fost gasita în date.")

# Salvam rezultatele într-un fișier nou pentru analiza ulterioara
data.to_csv('stroke_data_with_outliers.csv', index=False)
print("\nRezultatele au fost salvate in 'stroke_data_with_outliers.csv'. Graficele au fost salvate ca PNG.")

# ---
# Pentru rulare am folosit  pandas, numpy, scikit-learn, scipy
# pip install pandas numpy scikit-learn scipy matplotlib
# ---
