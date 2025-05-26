## Cerinte generale

1. La final se va preda codul sursa împreună cu fișierul de requirements.txt. 
<br>NU se va transmite și folderul venv.
2. Folosiți comentarii în interiorul programului pentru descrierea funcționalității.
3. Citați sursele folosite. Dacă ați reutilizat cod păstrați în comentariu de documentare de la începutul fișierului un link către codul original.
4. NU este necesară pentru prezentarea proiectelor o prezentare powerpoint. Codul sursă și un scurt demo sunt suficiente.
5. Totuși ar fi bine să vă pregătiți un discurs . Prezentarea poate fi realizată de un singur membru al echipei, dar ar fi indicat să fie toți membri implicați.
6. Toți membri echipei trebuie să fie capabili să răspundă la întrebări.
---

**Echipa**: 21-E6  
**Studenti**: DEJICA P. ANDREI-RADU  
**Tema proiect**: D7-T2 | Detecția anomaliilor 

Aplicați diferite tehnici pentru detecție de outliers în cadrul setului de date Cerebral Stroke Prediction-Imbalanced 
Dataset (kaggle.com).
https://www.kaggle.com/datasets/shashwatwork/cerebral-stroke-predictionimbalaced-dataset

Cerințe minimale:
1. Descărcați, citiți și încărcați într-un pandas DataFrame conținutul setului de date.
2. Aplicați metoda z-score pentru a identifica înregistrările de tip outliers la nivel de coloane, doar pentru 
coloanele numerice.
3. Aplicați modelul isolation forest pentru a descoperi înregistrări clasificate ca fiind outliers. Puteți folosi clasa 
minoritara pentru a vedea cât de bine reușește modelul să clasifice înregistrările considerate anormale.

**Resurse**:  
https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html#sklearn.ensemble.IsolationForest  
https://medium.com/@corymaklin/isolation-forest-799fceacdda4   