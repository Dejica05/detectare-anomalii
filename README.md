# Proiect D7-T2: Detectia Anomaliilor in Seturi de Date Medicale

**Echipa:** 21-E6  
**Student:** Dejica P. Andrei-Radu  
**Sursa de date:** [Cerebral Stroke Prediction – Kaggle](https://www.kaggle.com/datasets/shashwatwork/cerebral-stroke-predictionimbalaced-dataset)

---

## 1. Introducere

Acest proiect are ca obiectiv analiza unui set de date medicale care contine informatii despre pacienti si riscul acestora de a suferi un accident vascular cerebral (AVC). Scopul principal este identificarea valorilor anormale (outlieri), care pot semnala atat erori de colectare, cat si cazuri rare cu relevanta clinica sau statistica.

---

## 2. Metodologie

Au fost implementate doua metode complementare pentru detectia anomaliilor:

### 2.1. Detectie bazata pe scorul Z (Z-Score)

- Aplicata pe coloanele numerice ale setului de date.
- Se calculeaza scorul Z pentru fiecare valoare, folosind media si abaterea standard.
- Valorile cu `|z| > 3` sunt considerate outlieri.
- Rezultatele sunt vizualizate printr-un grafic de tip bara, ce indica distributia outlierilor pe fiecare coloana numerica.

### 2.2. Detectie bazata pe algoritmul *Isolation Forest*

- Aplicata pe aceleasi coloane numerice, completate pentru valorile lipsa.
- Se foloseste algoritmul `IsolationForest` din biblioteca `scikit-learn` pentru detectarea valorilor anormale intr-un mod nesupervizat.
- Fiecare observatie este clasificata ca **normala (1)** sau **outlier (-1)**.
- Se adauga o coloana suplimentara in dataset, denumita `is_outlier_iso`.
- Rezultatele sunt prezentate vizual intr-un grafic cu distributia outlierilor detectati.

---

## 3. Analiza Comparativa

Daca setul de date contine coloana `stroke` (indicand prezenta AVC), se realizeaza o analiza de tip tabel de contigenta (`crosstab`) intre valorile acestei coloane si etichetele generate de algoritmul Isolation Forest. Scopul este de a observa daca exista o corelatie intre cazurile de AVC si prezenta anomaliilor detectate.

---

## 4. Rezultate Salvate

- **`stroke_data_with_outliers.csv`** – contine setul de date initial, impreuna cu etichetele generate de algoritmul Isolation Forest.
- **`outliers_zscore_pe_coloana.png`** – grafic cu numarul de outlieri pe coloane, conform metodei Z-score.
- **`outliers_isoforest.png`** – grafic cu distributia valorilor normale si a outlierilor conform Isolation Forest.

---

## 5. Cerinte Tehnice

### Biblioteci utilizate:
- `pandas`
- `numpy`
- `scikit-learn`
- `scipy`
- `matplotlib`

### Instalare:
```bash
pip install pandas numpy scikit-learn scipy matplotlib
```

---

## 6. Instructiuni de Utilizare

1. Asigurati-va ca fisierul `dataset.csv` se afla in acelasi director cu scriptul Python.
2. Rulati scriptul folosind un mediu de dezvoltare Python (ex: terminal, Jupyter Notebook, VS Code).
3. Rezultatele vor fi afisate in consola si salvate automat in fisierele `.csv` si `.png`.

---

## 7. Concluzii

Detectia anomaliilor este un pas esential in analiza seturilor de date, mai ales in domeniul medical. Valorile aberante pot reprezenta atat erori, cat si situatii clinice deosebite. Prin utilizarea combinata a metodelor statistice (Z-score) si a celor bazate pe algoritmi de tip machine learning (Isolation Forest), se obtine o imagine mai clara si mai robusta asupra structurii si calitatii datelor analizate.
