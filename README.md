# Energy Loss Agent

**Un agent care prezice și reduce pierderile de energie la ambele capete ale lanțului: la producători și la consumatori.**

Proiect pentru **Smarthack XI 2026**, tema PPC *„24h to Power PPC: AI în energie, de la date la acțiune”*.

- **La producători** (turbine eoliene și parcuri fotovoltaice): estimăm cât ar trebui să producă fiecare echipament. Detectăm când produce sub normal și planificăm reviziile și controalele în ferestrele cu cea mai mică pierdere: vânt slab pentru turbine, soare puțin pentru panouri.
- **La consumatori** (locuințe): estimăm cât ar trebui să consume fiecare locuință. Detectăm anomaliile din seria lunară de consum (posibil furt sau contor defect) și prioritizăm inspecțiile.
- **Un agent LLM** pune toate pierderile într-o singură vedere (MWh și lei), explică fiecare decizie și rulează scenarii „ce-ar fi dacă”. Deciziile vin din modelele și optimizatorul nostru, nu din LLM.

---

## Cuprins

1. [Problema](#1-problema)
2. [Arhitectura](#2-arhitectura)
3. [Modulele](#3-modulele)
4. [Rezultate](#4-rezultate)
5. [Date](#5-date)
6. [Instalare și rulare](#6-instalare-și-rulare)
7. [Structura repository-ului](#7-structura-repository-ului)
8. [Evaluare](#8-evaluare)
9. [Robustețe și explicabilitate](#9-robustețe-și-explicabilitate)
10. [Limitări](#10-limitări)
11. [Echipa](#11-echipa)
12. [Licențe și citări](#12-licențe-și-citări)

---

## 1. Problema

Pierderile de energie apar în două locuri pe care PPC le acoperă cu divizii diferite:

| Unde | Ce se pierde | De ce contează pentru PPC |
|---|---|---|
| **Producție** (PPC Renewables) | Energie neprodusă când turbinele sau panourile merg sub potențial sau sunt oprite pentru mentenanță în momente cu vânt sau soare mult | Portofoliu de peste 1,5 GW în România, inclusiv parcul eolian Fântânele-Cogealac (600 MW, 240 de turbine GE 2.5xl) și parcul fotovoltaic Lumina, Călugăreni (63 MW) |
| **Distribuție** (Rețele Electrice) | Energie consumată, dar nefacturată: furt, contoare defecte | Un studiu din 2018 estima sustragerile la aprox. 3% din energia intrată în rețelele de distribuție; Rețele Electrice are 2 milioane de contoare inteligente |

Programul AI al grupului PPC include explicit mentenanța predictivă și reducerea pierderilor din rețea. Proiectul nostru pune cele două probleme sub un singur agent, cu o singură unitate de măsură: energia pierdută.

---

## 2. Arhitectura

Trei straturi, fiecare cu treaba lui:

```
STRAT 1 — MODELE ML (unități fizice + incertitudine)
  Producători                             Consumatori
  M1 producția așteptată (kW)             C1 consumul așteptat (kWh/lună)
  M2 funcționare sub normal (MWh/zi)      C2 anomalie: P(furt) + kWh pierduți/lună
  M3 risc de defect (opțional)
  M4 costul opririi pe oră (MWh/lei)

STRAT 2 — DECIZIE (determinist, reproductibil)
  Optimizator mentenanță                  Coada de inspecții
  (revizii, reparații, ferestre meteo,    (P(furt) × kWh, echipe pe zi,
   echipe, limite de vânt)                 grupare geografică)

STRAT 3 — AGENT LLM (coordonare + interfață)
  vedere unică a pierderilor · explicații · scenarii „ce-ar fi dacă”
  rapoarte de teren -> date structurate · ordine de lucru · alerte
```

```
SCADA / invertoare ─► M1 ─► M2 ─┐
prognoză meteo ─────► M1 ─► M4 ─┼─► Optimizator ──────┐
revizii + echipe ───────────────┘                      ├─► Agent LLM ─► calendar, inspecții,
consum lunar ───────► C1 ─► C2 ───► Coada de inspecții ┘                vedere pierderi, alerte
```

**Principiu:** LLM-ul nu calculează și nu decide. Orice număr afișat vine dintr-o unealtă, iar orice decizie poate fi urmărită până la predicția modelului care a produs-o.

---

## 3. Modulele

### 3.1 Producători: producția așteptată și planificarea mentenanței

| Piesă | Ce face | Algoritm | Baseline |
|---|---|---|---|
| **M1** Producția așteptată | Cât ar produce turbina sau invertorul acum, fără probleme | LightGBM pe cuantile (10/50/90%), antrenat doar pe perioade normale | Curba de putere pe intervale de 0,5 m/s (eolian); putere nominală × iradianță cu corecție de temperatură (PV) |
| **M2** Funcționare sub normal | Detectează pierderea persistentă, cauza probabilă și MWh pierduți pe zi | Indice de performanță pe 72 h + CUSUM + clasificator de cauză (LightGBM) antrenat pe defecte injectate | Prag fix pe indicele de performanță |
| **M3** Risc de defect *(opțional)* | Probabilitatea de defect în zilele următoare, din temperaturi anormale | Model de comportament normal pe temperaturi (CARE to Compare) | Prag fix pe temperatură; EnergyFaultDetector (Fraunhofer IEE) |
| **M4** Costul opririi | MWh (sau lei) pierduți dacă echipamentul e oprit în fiecare oră din următoarele 7–10 zile | M1 aplicat pe prognoza meteo | – |
| **Optimizator** | Alege ziua și echipa pentru fiecare revizie și reparație | Greedy (opțional CP-SAT) pe costul: oprire + pierdere cât aștepți + risc | Politici: „repar imediat”, „repar la revizia următoare” |

Cauzele detectate de M2: abatere de direcție (yaw), problemă de pitch, anemometru defect, givraj (eolian); murdărire, string căzut, umbrire, invertor limitat (PV).

### 3.2 Consumatori: anomalii în consumul lunar

| Piesă | Ce face | Algoritm | Baseline |
|---|---|---|---|
| **C1** Consumul așteptat | Cât ar trebui să consume locuința într-o lună | LightGBM pe cuantile, din istoricul propriu pe aceeași lună, tendință, locuințe similare, temperatură | Mediana aceleiași luni din anii anteriori × factor de tendință |
| **C2** Anomalie de consum | Probabilitatea de furt, kWh pierduți pe lună, tipul semnalului (furt / contor defect / nesigur) | Isolation Forest (nesupravegheat) + LightGBM (supravegheat, pe etichete SGCC și atacuri simulate), calibrat | Regula „consum cu peste X% sub aceeași lună de anul trecut” |
| **Coada de inspecții** | Ce locuințe se verifică primele | Valoare = P(furt) × kWh pierduți × luni recuperabile − cost deplasare; primele K pe echipă, grupate geografic | Ordonare doar după P(furt) |

Agentul nu acuză pe nimeni: programează verificări. Contoarele probabil defecte merg la echipa de măsură, cazurile nesigure la verificare de la distanță.

### 3.3 Agentul LLM

| Modul | Unelte expuse |
|---|---|
| Producători | `get_weather_forecast`, `predict_expected_production`, `detect_underperformance`, `compute_cost_map`, `optimize_maintenance` |
| Consumatori | `predict_expected_consumption`, `score_households`, `build_inspection_list` |
| Comune | `loss_overview`, `explain(item_id)`, `update_from_field_report(text)`, `create_work_orders`, `send_alert` |

Exemple de întrebări:

- „Ce pierderi avem săptămâna asta?”
- „De ce turbina T17 vineri și nu mâine?”
- „Echipa 2 lipsește joi.” (agentul replanifică și arată diferența de cost)
- „De ce e locuința X pe lista de azi?”
- „Am verificat locuința X: contor defect, nu furt.” (agentul cere confirmare, închide cazul și salvează eticheta)

Fără cheie de API pentru LLM, aplicația rulează în **mod CLI determinist**, cu aceleași unelte.

---

## 4. Rezultate

> Toate cifrele de mai jos se generează cu `make eval` și sunt salvate în `results/`. Completați tabelul după rulare.

| Componentă | Metrică | Model | Baseline |
|---|---|---|---|
| M1 eolian (Kelmarsh, an de test) | MAE (% din nominal) / acoperire interval 10–90% | _TBD_ | _TBD_ |
| M1 PV (Kaggle, ultimele 7 zile) | MAE (% din nominal) | _TBD_ | _TBD_ |
| M2 eolian (defecte injectate) | detecție % / întârziere (zile) / alarme false pe turbină-lună / F1 cauză | _TBD_ | _TBD_ |
| M3 (CARE to Compare, parcul A) | scor CARE | _TBD_ | _TBD_ |
| C2 consumatori (SGCC lunar, clienți de test) | PR-AUC / precizie în primele K / kWh recuperați în primele K | _TBD_ | _TBD_ |
| **Backtest mentenanță (un an)** | MWh pierduți total | **agent:** _TBD_ | **repar imediat:** _TBD_ · **la revizie:** _TBD_ |

Graficele (curbe de putere, matrice de confuzie, backtest) sunt în `results/figures/`.

---

## 5. Date

Folosim doar date publice și sintetice. Datele **nu** sunt în repository: scriptul `make data` le descarcă.

### 5.1 Producători

| Set | Ce folosim | Licență | Sursă |
|---|---|---|---|
| Kelmarsh wind farm data (Cubico) | SCADA la 10 minute, 6 turbine Senvion MM92, 1–2 ani | CC-BY-4.0 | https://zenodo.org/records/16807551 |
| CARE to Compare (Fraunhofer IEE) | 95 de seturi SCADA cu defecte etichetate (pentru M3) | CC BY-SA 4.0 | https://zenodo.org/records/15846963 |
| Hill of Towie (RES / TRIG) *(opțional)* | SCADA + jurnal de alarme; upgrade-uri cu dată cunoscută | CC-BY-4.0 | https://zenodo.org/records/14870023 |
| Solar Power Generation Data | 2 centrale PV, 34 de zile, date pe invertor la 15 minute | vezi pagina Kaggle | https://www.kaggle.com/datasets/anikannal/solar-power-generation-data |
| PVDAQ (NREL) *(opțional)* | Serii lungi de producție PV | CC-BY | https://data.openei.org/submissions/4568 |
| Open-Meteo | Vreme istorică și prognoză (vânt, iradianță, temperatură, presiune) | CC BY 4.0, uz necomercial | https://open-meteo.com |

### 5.2 Consumatori

| Set | Ce folosim | Licență | Sursă |
|---|---|---|---|
| SGCC electricity theft dataset | 42.372 de clienți, consum zilnic 2014–2016, eticheta de furt; agregat lunar | vezi pagina setului | Kaggle, „SGCC Electricity Theft Detection” (bensalem14); descriere: https://baselight.app/u/kaggle/dataset/bensalem14_sgcc_dataset |
| Smart Meters in London | Consum la 30 de minute, agregat lunar, cu atacuri simulate | vezi pagina setului | https://www.kaggle.com/datasets/jeanmidev/smart-meters-in-london |
| Synthetic Data for Smart Meter Attack Detection *(verificare)* | 159 de utilizatori, 7 strategii de atac | vezi pagina setului | https://ieee-dataport.org/documents/synthetic-data-smart-meter-attack-detection |

### 5.3 Defecte și atacuri sintetice

Datele publice nu conțin etichete pentru toate cazurile, așa că le generăm controlat și documentat (`src/simulate/`):

- **producători:** abatere de direcție, pitch, anemometru, givraj, murdărire, string căzut, umbrire, invertor limitat;
- **consumatori:** funcțiile clasice de atac din literatură (înmulțire cu o constantă, constantă aleatoare, oprire pe intervale etc.), aplicate pe consum „onest”.

Toate injecțiile folosesc seed fix, iar etichetele sunt salvate separat.

> Weather data by [Open-Meteo.com](https://open-meteo.com/).

---

## 6. Instalare și rulare

**Cerințe:** Python 3.11, aprox. 5 GB spațiu liber pentru setul minim (peste 10 GB dacă descărcați și CARE to Compare).

```bash
git clone <url-repo>
cd <repo>
python -m venv .venv && source .venv/bin/activate
make setup
```

**Token Kaggle** (pentru seturile PV și de consum): creați `~/.kaggle/kaggle.json` din contul Kaggle → *Settings* → *API*.

**LLM** *(opțional)*: setați variabila de mediu pentru furnizorul ales (vezi `config/agent.yaml`). Fără ea, agentul rulează în mod CLI.

| Comandă | Ce face |
|---|---|
| `make data` | Descarcă setul minim (Kelmarsh, Kaggle solar, SGCC, vreme) și îl aduce la schema comună în `data/processed/` |
| `make data-full` | Descarcă și seturile opționale (CARE to Compare, Hill of Towie, PVDAQ) |
| `make train` | Antrenează M1, M2, C1, C2 (și M3 dacă datele există) |
| `make eval` | Evaluează toate modelele față de baseline și rulează backtest-ul; scrie `results/*.json` și graficele |
| `make agent` | Pornește API-ul agentului (FastAPI) |
| `make demo` | Pornește interfața (Streamlit) |
| `make test` | Rulează testele de schemă și de optimizator |

Toate ipotezele numerice (limita de vânt pentru lucru, costul unui defect, durata lucrărilor, capacitatea echipelor, prețul energiei) sunt în `config/params.yaml`.

---

## 7. Structura repository-ului

```
.
├── README.md
├── AGENTS.md                    # ghid pentru agentul de programare
├── Makefile
├── requirements.txt
├── config/
│   ├── params.yaml              # ipoteze numerice, documentate
│   ├── sites.yaml               # coordonate: Kelmarsh, Fântânele-Cogealac, Călugăreni
│   └── agent.yaml               # furnizor LLM, praguri de încredere
├── data/                        # (ignorat de git) raw/ interim/ processed/
├── models/                      # (ignorat de git)
├── results/                     # metrici JSON, grafice, ordine de lucru generate
├── docs/
│   ├── data_notes.md            # ce am găsit la inspecția datelor
│   └── decisions.md             # de ce am ales pragurile și parametrii
├── src/
│   ├── data/                    # descărcare, ingestie, vreme
│   ├── features/                # densitatea aerului, turbulență, poziția soarelui, profil lunar
│   ├── producers/
│   │   ├── m1_expected.py
│   │   ├── m2_underperf.py
│   │   ├── m3_risk.py
│   │   └── m4_cost.py
│   ├── consumers/
│   │   ├── c1_expected.py
│   │   ├── c2_anomaly.py
│   │   └── inspections.py
│   ├── simulate/                # defecte și atacuri sintetice
│   ├── decision/
│   │   ├── optimizer.py
│   │   └── backtest.py
│   ├── agent/
│   │   ├── tools.py
│   │   ├── llm_agent.py
│   │   └── api.py
│   └── app/
│       └── dashboard.py
└── tests/
```

---

## 8. Evaluare

- **Split-uri fără scurgeri de informație:**
  - serii de producție: împărțire după timp;
  - consumatori: împărțire după client (niciun client în train și test) și după timp;
  - setul de test e fixat la început.
- **Fiecare model e comparat cu un baseline simplu** (tabelele din secțiunea 3).
- **Producători:** MAE și acoperirea intervalelor (M1); detecție, întârziere, alarme false și F1 pe cauze (M2); scorul CARE (M3).
- **Consumatori:** PR-AUC, precizia în primele K și kWh recuperați în primele K, unde K = capacitatea zilnică de inspecție.
- **Decizia:** backtest pe un an. Agentul rulează zi cu zi, văzând doar datele de până atunci, cu prognoză perfectă și cu prognoză cu zgomot. Îl comparăm cu „repar imediat” și „repar la revizia următoare”.

Analiza erorilor (unde greșește fiecare model, pe ce tipuri de cazuri și ce am încercat) e în `docs/decisions.md` și în prezentare.

---

## 9. Robustețe și explicabilitate

**Robustețe:**

- golurile scurte se interpolează, cele lungi se marchează;
- senzorii blocați și valorile imposibile se filtrează;
- turbinele în service sau limitate sunt excluse din detecție;
- prognoză indisponibilă → ultimul cache, cu avertisment;
- intrări invalide în API → erori clare;
- încredere mică → agentul spune „nu sunt sigur” și trimite cazul la un om (verificare cu drona, verificare de la distanță a contorului).

**Explicabilitate:**

- valori SHAP pentru fiecare alarmă (primele 3 motive);
- pentru fiecare lucrare programată: descompunerea costului (oprire / așteptare / risc) și variantele respinse, cu motiv;
- pentru fiecare locuință suspectă: curba lunară față de așteptare și față de locuințele similare.

---

## 10. Limitări

- **Datele nu sunt de la PPC.** Turbinele (Senvion, Siemens) și centralele PV (India, SUA) diferă de cele din portofoliul PPC. La Fântânele-Cogealac și Călugăreni folosim doar vremea reală, pentru demonstrație.
- **Setul PV Kaggle are doar 34 de zile.** Rezultatele PV sunt orientative.
- **Defectele și atacurile sunt parțial sintetice.** Performanța pe cazuri reale poate fi mai mică; pentru furt, testăm separat pe etichetele reale SGCC.
- **Granularitatea lunară ascunde unele tipuri de furt** (de exemplu mutarea consumului între ore). Datele zilnice sau orare de la contoarele inteligente ar îmbunătăți detecția.
- **Prosumatorii noi** (consum net în scădere după montarea panourilor) pot arăta ca furt dacă atributul „prosumator” lipsește.
- **Parametrii operaționali sunt ipoteze** (limite de vânt, costuri, durate): în producție ar veni din manualele producătorilor și din datele PPC.

Pentru producție ar fi nevoie de la PPC de: SCADA reală, jurnal de service, contractele de service, tipul invertoarelor, istoricul inspecțiilor și rezultatele lor.

---

## 11. Echipa

| Membru | Rol |
|---|---|
| Valentin | Modulul producători: producția așteptată (eolian + PV), costul opririi, optimizatorul de mentenanță, backtest |
| _Coleg_ | Modulul consumatori: consumul așteptat, detecția anomaliilor, coada de inspecții |
| _Toată echipa_ | Agentul LLM, interfața, evaluarea, prezentarea |

---

## 12. Licențe și citări

**Cod:** MIT (vezi `LICENSE`).

**Date:** fiecare set își păstrează licența proprie (secțiunea 5). Citări:

- Cubico Sustainable Investments, *Kelmarsh wind farm data*, Zenodo. https://zenodo.org/records/16807551
- Gück, C., Roelofs, C. M. A., Faulstich, S. *CARE to Compare: A Real-World Benchmark Dataset for Early Fault Detection in Wind Turbine Data*. Data 2024, 9(12), 138. https://doi.org/10.3390/data9120138 · date: https://zenodo.org/records/15846963
- RES on behalf of TRIG, *Hill of Towie wind farm open dataset*, Zenodo. https://zenodo.org/records/14870023
- NREL, *Photovoltaic Data Acquisition (PVDAQ) Public Datasets*. https://doi.org/10.25984/1846021
- *Solar Power Generation Data*, Kaggle (anikannal).
- State Grid Corporation of China, *SGCC electricity theft dataset*, via Kaggle.
- *Smart Meters in London*, Kaggle (jeanmidev), date de la London Datastore / UK Power Networks.
- Open-Meteo.com Weather API, CC BY 4.0. https://open-meteo.com
- Fraunhofer IEE, *EnergyFaultDetector*. https://pypi.org/project/energy-fault-detector/
- pvlib-python. https://pvlib-python.readthedocs.io
