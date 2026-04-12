# 🧩 RL Labyrinth — Algorithm Comparison

> **EN:** A visual side-by-side comparison of Reinforcement Learning and classical pathfinding algorithms on randomly generated mazes.
>
> **CZ:** Vizuální porovnání Reinforcement Learning a klasických algoritmu pro hledani cesty na nahodne generovanem bludisti.

Built with **Python & Pygame**, this tool lets you watch 6 algorithms race through the same maze in real-time, collect detailed JSON results, and generate publication-ready charts with a single command.

Postaveno v **Pythonu a Pygame**. Aplikace umoznuje sledovat 6 algoritmu na stejnem bludisti, ukladat JSON vysledky a tvorit grafy jednim prikazem.

---

## 📑 Table of Contents

- [✨ Features / Funkce](#-features--funkce)
- [🖼️ Algorithms / Algoritmy](#️-algorithms--algoritmy)
- [🚀 Getting Started / Rychly start](#-getting-started--rychly-start)
  - [Prerequisites / Pozadavky](#prerequisites--pozadavky)
  - [Installation / Instalace](#installation--instalace)
  - [Running the Program / Spusteni](#running-the-program--spusteni)
- [🎮 How to Use (EN/CZ)](#-how-to-use-encz)
  - [Mode Selection / Vyber rezimu](#1️⃣-mode-selection--vyber-rezimu)
  - [Record EVERY Algorithm (Comparison Mode) / Porovnani vsech algoritmu](#2️⃣-record-every-algorithm-comparison-mode--porovnani-vsech-algoritmu)
  - [Record ONE Algorithm / Jediny algoritmus](#3️⃣-record-one-algorithm--jediny-algoritmus)
  - [Multi-Run Batch Mode / Davkove behy](#4️⃣-multi-run-batch-mode--davkove-behy)
  - [Parameter Sweep / Parametricky sweep](#5️⃣-parameter-sweep--parametricky-sweep)
- [⌨️ Keyboard Controls / Ovladani](#️-keyboard-controls--ovladani)
- [📊 Visualization Script / Vizualizacni skript](#-visualization-script--vizualizacni-skript)
- [📁 Project Structure / Struktura projektu](#-project-structure--struktura-projektu)
- [📄 Output Files / Vystupni soubory](#-output-files--vystupni-soubory)
- [⚙️ Configuration / Nastaveni](#️-configuration--nastaveni)
- [📝 License / Licence](#-license--licence)

---

## ✨ Features / Funkce

| Feature | Description (EN) | Popis (CZ) |
|---------|------------------|------------|
| 🏁 **6 Algorithms Side-by-Side** | Q-Learning, SARSA, BFS, Wall Follower, Random Walk, Greedy — all on the same maze | Q-Learning, SARSA, BFS, Wall Follower, Random Walk, Greedy — vse na stejnem bludisti |
| 🎲 **Random Maze Generation** | Each run creates a unique random maze from your size/config settings | Kazdy beh vytvori nove nahodne bludiste podle nastaveni |
| 📈 **Multi-Run Batching** | Run 1–500+ comparisons automatically, all saved in one JSON | Davkove porovnani 1–500+ behu do jednoho JSON |
| 🖥️ **Live 2×3 Grid View** | Watch all algorithms train/solve simultaneously | 2×3 mrizka se soucasnym treningem/resenim |
| 🔍 **Click to Expand** | Click any algorithm panel to view it fullscreen, press **B** to go back | Klik pro full screen, **B** pro navrat do mrizky |
| 📊 **Auto Chart Generation** | 10 single-run charts or 9 multi-run charts (box plots, learning curves, radar, etc.) | 10 grafu pro single-run nebo 9 pro multi-run |
| ⚡ **Speed Controls** | 1×–1000× speed multiplier (keys **1**–**0**) | Zrychleni 1×–1000× klavesami **1**–**0** |
| 🗺️ **Optimal Path Overlay** | Toggle shortest path arrows with **P** | Prepinani optimalni cesty klavesou **P** |
| 💾 **Comprehensive JSON Output** | Paths, heatmaps, Q-tables, success rates, timings | Cesty, heatmapy, Q-tabulky, uspesnost, casy |
| 🔄 **Resizable Window** | UI scales to any window size | UI se prispusobi velikosti okna |
| 🧪 **Parameter Sweep** | Built-in parameter sweep tool from the mode screen | Vestaveny parametricky sweep z uvodni obrazovky |

---

## 🖼️ Algorithms / Algoritmy

### 🤖 Reinforcement Learning

**EN:** RL algorithms use ε-greedy exploration with exponential decay for ε and α.

**CZ:** RL algoritmy pouzivaji ε-greedy pruzkum s exponencialnim snizovanim ε a α.

| Algorithm | Strategy / Strategie | Update Rule / Aktualizace |
|-----------|----------|-------------|
| **Q-Learning** | Off-policy, learns optimal Q-values directly | $Q(s,a) \leftarrow Q(s,a) + \alpha [r + \gamma \max_{a'} Q(s',a') - Q(s,a)]$ |
| **SARSA** | On-policy, learns from actions actually taken | $Q(s,a) \leftarrow Q(s,a) + \alpha [r + \gamma Q(s',a') - Q(s,a)]$ |

### 🧭 Classical (Non-RL)

| Algorithm | Guarantees Shortest Path? / Zarucuje nejkratsi cestu? | Deterministic? / Deterministicky? | Uses Heuristic? / Heuristika? |
|-----------|:--------------------------------------------------------:|:---------------------------------:|:-------------------------------:|
| **BFS** (Breadth-First Search) | ✅ | ✅ | ❌ |
| **Wall Follower** (Right-Hand Rule) | ❌ | ✅ | ❌ |
| **Random Walk** | ❌ | ❌ | ❌ |
| **Greedy Best-First** (Manhattan) | ❌ | ✅ | ✅ |

---

## 🚀 Getting Started / Rychly start

### Prerequisites / Pozadavky

- **Python 3.10+** (3.11 nebo 3.13 doporuceno)
- **pip** (soucast instalace Pythonu)

### Installation / Instalace

**EN: 1) Clone the repository**

```bash
git clone https://github.com/SpootyIsBest/RL_Labyrinth_-_Algorithm_Comparison_AP.git
cd RL_Labyrinth_-_Algorithm_Comparison_AP
```

**CZ: 1) Naklonujte repozitar**

```bash
git clone https://github.com/SpootyIsBest/RL_Labyrinth_-_Algorithm_Comparison_AP.git
cd RL_Labyrinth_-_Algorithm_Comparison_AP
```

**EN: 2) (Optional) Create a virtual environment**

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

**CZ: 2) (Volitelne) Vytvorte virtualni prostredi**

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

**EN: 3) Install dependencies**

```bash
pip install -r requirements.txt
```

This installs: `pygame`, `numpy`, `matplotlib`, `imageio`, `imageio-ffmpeg`, `plotly`

**CZ: 3) Nainstalujte zavislosti**

```bash
pip install -r requirements.txt
```

Nainstalujete: `pygame`, `numpy`, `matplotlib`, `imageio`, `imageio-ffmpeg`, `plotly`

### Running the Program / Spusteni

```bash
python main.py
```

**EN:** A Pygame window (1280×720, resizable) opens at the **Mode Selection** screen.

**CZ:** Otevre se Pygame okno (1280×720, resizable) na obrazovce **Mode Selection**.

---

## 🎮 How to Use (EN/CZ)

### 1️⃣ Mode Selection / Vyber rezimu

**EN:** When the program starts, choose a mode:

| Button | What it does |
|--------|-------------|
| **Record ONE Algorithm** | Train/visualize a single RL or Non-RL algorithm |
| **Record EVERY Algorithm** | Run all 6 algorithms on the same maze in parallel *(recommended)* |
| **Parameter Sweep** | Run the parameter sweep tool in a separate window |

**CZ:** Po spusteni programu vyberte rezim:

| Tlacitko | Co dela |
|----------|--------|
| **Record ONE Algorithm** | Trenink/vizualizace jedne RL nebo Non-RL metody |
| **Record EVERY Algorithm** | Soubezny beh vsech 6 algoritmu na stejnem bludisti *(doporučeno)* |
| **Parameter Sweep** | Spusti parametricky sweep v samostatnem okne |

---

### 2️⃣ Record EVERY Algorithm (Comparison Mode) / Porovnani vsech algoritmu

**EN:** Main workflow:

```
Mode Selection → Comparison Setup → RL Settings → Comparison Visualization → Results
```

**Step 1 — Comparison Setup Menu**
- Set **maze name**, **description**, **maze size** (width × height), **number of episodes**, **steps per episode**, and **rewards**
- Click **Continue to RL Settings**

**Step 2 — RL Settings**
- Configure RL hyperparameters (ε, α, γ and their decay rates)
- Set **Number of Runs** at the bottom (default `1`, use higher for batch mode)
- Click **Continue to Load Maze**

**Step 3 — Load Maze**
- Maze is generated from the comparison setup JSON
- The app enters the comparison visualization automatically

**Step 4 — Comparison Visualization**
- A random maze is generated and all **6 algorithms start simultaneously**
- The screen shows a **2×3 grid** with each algorithm's live progress
- RL algorithms show training heatmaps; Non-RL algorithms show step-by-step solving
- 💡 **Press F to freeze visualization** — training runs 10–100× faster in the background
- Use keys **1**–**0** to adjust speed (1×–1000×)

**Step 5 — Results**
- When all algorithms finish, press **R** to see the results overlay
- Results are auto-saved as JSON
- Press **Q** to return to the main menu

**CZ:** Hlavni postup:

```
Mode Selection → Comparison Setup → RL Settings → Comparison Visualization → Results
```

**Krok 1 — Comparison Setup Menu**
- Nastavte **nazev bludiste**, **popis**, **velikost bludiste** (sirka × vyska), **pocet epizod**, **kroky na epizodu** a **odmeny**
- Kliknete **Continue to RL Settings**

**Krok 2 — RL Settings**
- Nastavte RL hyperparametry (ε, α, γ a jejich decay hodnoty)
- Dole zadejte **Number of Runs** (vychozi `1`, vyssi hodnota = batch)
- Kliknete **Continue to Load Maze**

**Krok 3 — Load Maze**
- Bludiste se vytvori z porovnani JSON
- Aplikace automaticky prejde do vizualizace

**Krok 4 — Comparison Visualization**
- Vygeneruje se nahodne bludiste a vsech **6 algoritmu startuje soucasne**
- Obrazovka zobrazuje **2×3 mrizku** s prubehem kazdeho algoritmu
- RL ukazuje treningove heatmapy; Non-RL ukazuje krokove reseni
- 💡 **Stisknete F pro zmrazeni vizualizace** — trening bezi na pozadi rychleji
- Klavesami **1**–**0** nastavite rychlost (1×–1000×)

**Krok 5 — Results**
- Po dokonceni stisknete **R** pro zobrazeni vysledku
- Vysledky se automaticky ulozi do JSON
- **Q** vrati do hlavniho menu

---

### 3️⃣ Record ONE Algorithm / Jediny algoritmus

**EN:** For testing a single algorithm:

- **RL path:** Choose RL Algorithms → Create New Maze or Load Existing Maze → RL Settings → click **Start Training** → in visualization press **S** to start/stop
- **Non-RL path:** Choose Non-RL Algorithms → Pick algorithm from dropdown → Continue → Set speed → click **Start Visualization**

**CZ:** Pro test jedineho algoritmu:

- **RL cesta:** Vyberte RL Algorithms → Create New Maze nebo Load Existing Maze → RL Settings → kliknete **Start Training** → ve vizualizaci stisknete **S**
- **Non-RL cesta:** Vyberte Non-RL Algorithms → Vyberte algoritmus z dropdown → Continue → Nastavte rychlost → kliknete **Start Visualization**

---

### 4️⃣ Multi-Run Batch Mode / Davkove behy

**EN:** For statistically meaningful comparisons across many random mazes:

1. Go to **Record EVERY Algorithm** → Comparison Setup → RL Settings
2. In the **"Number of Runs"** input at the bottom, enter a number (e.g. `100`)
3. Click continue — the system will:
  - Run a comparison on a random maze
  - When all 6 algorithms finish, **automatically generate a new maze**
  - Repeat until all runs are complete
4. All results are saved in **one JSON file** with per-run data and aggregate statistics
5. The status bar shows **"Run X/N"** in orange during the batch

> 💡 **Tip:** Press **F** to freeze visualization and **0** for 1000× speed to blast through hundreds of runs quickly.

**CZ:** Pro statisticky vyznamne porovnani pres mnoho nahodnych bludist:

1. Jdete na **Record EVERY Algorithm** → Comparison Setup → RL Settings
2. V poli **"Number of Runs"** dole zadejte cislo (napr. `100`)
3. Kliknete continue — system:
  - Spusti porovnani na nahodnem bludisti
  - Po dokonceni vsech 6 algoritmu **automaticky vygeneruje nove bludiste**
  - Opakuje, dokud nedobehnou vsechny behy
4. Vse se ulozi do **jedineho JSON souboru** s daty po behu i agregacemi
5. Stavovy radek ukazuje **"Run X/N"** (oranzove) behem batchu

> 💡 **Tip:** Stisknete **F** pro zmrazeni vizualizace a **0** pro 1000× rychlost.

---

### 5️⃣ Parameter Sweep / Parametricky sweep

**EN:** On the Mode Selection screen, click **Parameter Sweep**. This launches `parameter_sweep.py` in a separate process. Use it to run grid sweeps over RL hyperparameters and save results.

**CZ:** Na obrazovce Mode Selection kliknete **Parameter Sweep**. Spusti se `parameter_sweep.py` v samostatnem procesu. Slouzi k parametrickym sweepum RL hyperparametru a ukladu vysledku.

---

## ⌨️ Keyboard Controls / Ovladani

### 🏁 During Comparison Visualization

| Key | Action (EN) | Akce (CZ) |
|-----|-------------|-----------|
| **1–9, 0** | Speed: 1×, 2×, 5×, 10×, 20×, 50×, 100×, 200×, 500×, 1000× | Rychlost: 1×, 2×, 5×, 10×, 20×, 50×, 100×, 200×, 500×, 1000× |
| **F** | Freeze/unfreeze visualization (training continues) | Zmrazeni/obnoveni vizualizace (trening bezi) |
| **T** | Pause/resume training computation | Pozastavit/pokracovat trening |
| **P** | Toggle optimal path arrows overlay | Prepnout optimalni cestu |
| **C** | Toggle settings overlay | Prepnout overlay s nastavenim |
| **R** | Toggle results overlay (after completion) | Prepnout vysledky (po dokonceni) |
| **Q** | Quit to main menu (from results overlay) | Navrat do hlavniho menu (z vysledku) |
| **Click** | Expand an algorithm panel to fullscreen | Rozsirit panel na celou obrazovku |
| **B** | Return to 2×3 grid from expanded view | Navrat do 2×3 mrizky |
| **ESC** | Exit the application | Ukoncit aplikaci |

### 🤖 During Single RL Visualization

| Key | Action (EN) | Akce (CZ) |
|-----|-------------|-----------|
| **S** | Start / stop training | Spustit / zastavit trening |
| **Q** | Toggle Q-value display | Prepnout zobrazeni Q-hodnot |
| **I** | Toggle optimal path overlay | Prepnout optimalni cestu |
| **V** | Toggle config overlay | Prepnout overlay s konfiguraci |

### 🧭 During Single Non-RL Visualization

| Key | Action (EN) | Akce (CZ) |
|-----|-------------|-----------|
| **Space** | Pause / resume | Pauza / pokracovat |
| **I** | Toggle optimal path overlay | Prepnout optimalni cestu |

---

## 📊 Visualization Script / Vizualizacni skript

After running comparisons, generate charts from the saved JSON:

```bash
# Auto-pick newest JSON, show interactively
python visualize_comparison.py

# Use a specific file
python visualize_comparison.py my_results.json

# Save all charts as PNGs
python visualize_comparison.py --save

# Generate every chart type
python visualize_comparison.py --all
```

**EN:** The script auto-detects single-run vs multi-run format. It reads JSONs from `ComparisonVisualization/` by default.

**CZ:** Skript automaticky rozlisi single-run a multi-run format. Standardne nacita JSON z `ComparisonVisualization/`.

### Single-Run Charts (10)

| # | Chart | Description |
|---|-------|-------------|
| 1 | 📋 Summary Dashboard | 7-panel overview of all key metrics |
| 2 | 📏 Path Lengths | Bar chart with optimal path baseline |
| 3 | ⏱️ Execution Times | Training / solving time comparison |
| 4 | 🎯 RL Success Rates | Success rate + first successful episode |
| 5 | 📈 Learning Curves | Success rate over episode windows |
| 6 | 🔥 Heatmaps | Side-by-side exploration frequency maps |
| 7 | ⚡ Path Efficiency | Efficiency ratio + extra steps |
| 8 | 🗺️ Exploration Coverage | Percentage of maze explored |
| 9 | 📋 Non-RL Table | Property comparison table |
| 10 | 🕸️ Radar Chart | Normalized multi-metric spider chart |

### Multi-Run Charts (9)

| # | Chart | Description |
|---|-------|-------------|
| 1 | 📋 Multi-Run Dashboard | Overview with box plots, curves, win rates |
| 2 | 📦 Path Length Box Plots | Distribution across all runs |
| 3 | 📦 Execution Time Box Plots | Time distribution per algorithm |
| 4 | 📦 RL Success Rate Box Plots | Success rate distribution with data points |
| 5 | 📈 Averaged Learning Curves | Mean ± std deviation shaded bands |
| 6 | 📦 Efficiency Box Plots | Path efficiency distribution |
| 7 | 📊 Aggregate Statistics | Average path length with min/max error bars |
| 8 | 🏆 Win Rate | How often each algorithm found the shortest path |
| 9 | 🎯 Consistency | Coefficient of variation (lower = more consistent) |

Charts are saved to `ComparisonVisualization/charts/`.

---

## 📁 Project Structure / Struktura projektu

```
📂 RL_Labyrinth_-_Algorithm_Comparison_AP/
│
├── 🎮 main.py                     # Main entry point (Pygame application)
├── 🤖 Agent.py                    # Agent logic & policy updates
├── 🧩 Maze.py                     # Maze environment & transitions
├── 📍 State.py                    # State representation
├── 🧭 NonRL_Algorithms.py         # BFS, Wall Follower, Random Walk, Greedy
├── 🎨 NonRL_Visualizer.py         # Step-by-step Non-RL visualizer
├── 📺 Monitors.py                 # Screen/monitor management
├── 🔘 Button.py                   # UI button component
├── 🔄 Button_On_Off.py            # Toggle button component
├── 📝 InputBox.py                 # Text input box component
├── 📋 Drop_Down_Menu.py           # Dropdown menu component
├── 📋 Algorithm_Dropdown.py       # Algorithm selector dropdown
├── 📊 visualize_comparison.py     # Post-hoc chart generator (matplotlib)
├── 🧪 parameter_sweep.py           # Parameter sweep tool
├── 🔥 heatmap_gif.py              # Heatmap animation generator
├── 👁️ heatmap_viewer.py           # Heatmap viewer utility
│
├── 📄 default_data.json           # Template for single-algorithm mode
├── 📄 default_comparison_data.json# Template for comparison mode
├── 📄 default_comparison_results.json # Template for comparison output
├── 📄 default_layout.json         # Default maze layout
├── 📄 default_nonrl_data.json     # Template for Non-RL results
├── 📄 requirements.txt            # Python dependencies
│
├── 📂 JsonData/                   # Saved maze configurations
├── 📂 MazeLayouts/                # Saved maze wall layouts
├── 📂 NonRL_Results/              # Comparison & multi-run result JSONs
├── 📂 ComparisonVisualization/    # Results for chart generation
│   └── 📂 charts/                 # Generated PNG charts
├── 📂 HeatMaps_vids/              # Heatmap animations
└── 📂 oldCode/                    # Archived legacy implementations
```

---

## 📄 Output Files / Vystupni soubory

| File | Location | When Created (EN) | Kdy se vytvari (CZ) |
|------|----------|------------------|---------------------|
| Single-run comparison JSON | `NonRL_Results/Comparison_<maze>_<timestamp>.json` | After each comparison finishes | Po dokonceni porovnani |
| Multi-run batch JSON | `NonRL_Results/MultiRun_<maze>_<Nruns>runs_<timestamp>.json` | After batch completes | Po dokonceni davky |
| Single Non-RL JSON | `NonRL_Results/<maze>_<algorithm>_<timestamp>.json` | After single Non-RL run | Po behu jednoho Non-RL algoritmu |
| RL training JSON | `JsonData/<name>.json` | After RL training ends | Po dokonceni RL treningu |
| Chart PNGs | `ComparisonVisualization/charts/` | When running `visualize_comparison.py --save` | Pri spusteni `visualize_comparison.py --save` |
| Maze layout | `MazeLayouts/<maze>_layout.json` | When saving a maze layout | Pri ulozeni layoutu |
| Comparison copy | `ComparisonVisualization/*.json` | Auto-copied after comparisons | Automaticky po porovnani |

---

## ⚙️ Configuration / Nastaveni

### Maze Settings (Comparison Setup Menu) / Nastaveni bludiste

| Parameter | Description (EN) | Popis (CZ) | Example |
|-----------|------------------|-----------|---------|
| name | Identifier for the maze | Nazev bludiste | `MyTest` |
| MazeSize | Width × Height in cells | Sirka × vyska v bunkach | `[25, 25]` |
| NumOfEpisodes | Number of RL training episodes | Pocet epizod treninku | `625` |
| stepsPerEpisode | Max steps before episode ends | Max kroku na epizodu | `313` |
| rewardForFinish | Reward when reaching the goal | Odmena za cil | `50` |
| rewardForValidMove | Reward (penalty) per step | Odmena (pokuta) za krok | `-1` |

### RL Hyperparameters (RL Settings Menu) / RL hyperparametry

| Parameter | Description (EN) | Popis (CZ) | Default |
|-----------|------------------|-----------|---------|
| ε₀ (Epsilon Start) | Initial exploration rate | Pocatecni mira pruzkumu | `0.9` |
| ε_min (Epsilon Min) | Minimum exploration rate | Minimalni mira pruzkumu | `0.05` |
| ε_decay | Epsilon decay per episode | Decay ε na epizodu | `0.9995` |
| α₀ (Alpha Start) | Initial learning rate | Pocatecni learning rate | `0.72` |
| α_min (Alpha Min) | Minimum learning rate | Minimalni learning rate | `0.1` |
| α_decay | Alpha decay per episode | Decay α na epizodu | `0.997` |
| γ (Gamma) | Discount factor | Diskontni faktor | `1.0` |

### Other Settings / Dalsi nastaveni

**EN:** The **Settings** screen includes a toggle for heatmap recording and a **Save as Layout** button (saves to `MazeLayouts/`).

**CZ:** Obrazovka **Settings** obsahuje prepinac ukladani heatmap a tlacitko **Save as Layout** (ulozi do `MazeLayouts/`).

---

## 📝 License / Licence

No license specified at this time.
