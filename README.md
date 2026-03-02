# Mini-Project-1A
# Emotion Classification with Brain Circuit Simulation

## Project Description

This project simulates the amygdala (fear circuit) response to threatening stimuli using a Python-based neural dynamics model. Users adjust input parameters to control stimulus intensity and neural properties, and the model predicts the brain state (fear or no fear) based on oscillation frequency. The system demonstrates a simple end-to-end CI workflow that runs identically on both local machines and FABRIC distributed computing platforms.

## Dataset Source

No external dataset required. The simulation generates synthetic threat stimulus signals (Gaussian curves) and neural responses based on mathematical parameters.

## Environment Setup

**Using Conda:**
```bash
conda create -n fear_sim python=3.9
conda activate fear_sim
pip install -r requirements.txt
```

**Using Pip:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## How to Run

**Basic run with default parameters:**
```bash
python fear_simulation.py
```

**Run with custom parameters:**
```bash
python fear_simulation.py --stimulus-intensity 0.8 --threshold 0.4 --duration 30
```

**Run on FABRIC:**
Clone the repository on FABRIC, install dependencies, and run the same command. The code works identically on both platforms.

**Available parameters:**
- `--stimulus-intensity`: Threat level (0-1, default 0.3)
- `--threshold`: Neural firing threshold (0-1, default 0.5)
- `--neural-gain`: Neuron sensitivity (0.5-2.0, default 1.0)
- `--duration`: Simulation time in seconds (default 20)
- `--config`: Path to config file (default config.json)
- `--no-plot`: Skip plot generation
- `--no-csv`: Skip CSV data export

## Sample Output

Running with default parameters:
```
Running simulation for 20.0s...
  10%
  20%
  ...
  100%
Done!

============================================================
RESULTS
============================================================
Stimulus Intensity: 0.3
Firing Threshold: 0.5

Brain State: NO FEAR
Confidence: 85.43%
Oscillation Frequency: 7.23 Hz
Mean Firing Rate: 0.234
============================================================
```

The script generates:
- **amygdala_plot.png**: 3-panel visualization showing stimulus input, membrane potential, and firing rate
- **amygdala_data.csv**: Raw time-series data for analysis

**Test with high stimulus (fear response):**
```bash
python fear_simulation.py --stimulus-intensity 0.9 --threshold 0.4
```

Expected output: Brain State = FEAR, Oscillation Frequency = 25-35 Hz

**Test with low stimulus (calm state):**
```bash
python fear_simulation.py --stimulus-intensity 0.1
```

Expected output: Brain State = NO FEAR, Oscillation Frequency = 5-8 Hz
