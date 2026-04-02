# Anomaly classification using physics inspired LSTM

Most anomaly detection method aim to detect presence of an anomaly. 
This project goes one step beyond and explores how to detect the cause of the anomaly. 
We assume that a clean signal is available at the time of training.
The project is executed in the domain of time-series analysis and uses the LSTM architecture. 

---
![Anomaly classification in 2D MSE -- physics loss plane](https://github.com/suchitakulkarni/anomaly_classification/blob/main/results/2d_scatter_comparison.png)


## Core idea

The central idea is as follows. An anomaly originating from a physical cause, 
e.g., a glitch in the amplitude of oscillation, it breaks the underlying physics
model differently than a damped amplitude. If this change in the amount and the type of 
physics loss can be captured, one can identify the anomaly type e.g. amplitude damping vs. glitch. 

# Dataset
The framework uses simulated simple harmonic oscillator data as a controlled testbed for testing the formalism. .

---

## Key findings

- The physics informed approach performs better than the data driven appraoch in presence of noise.
- Adding physics constraints includes a tradeoff between reconstruction accuracy and physics compliance.
- The physics informed approach obtained 57% improvements over the data driven approach in terms of anomaly seperation.
- Lower physics weight leads to better detection of anomalies. 
---

## File structure

```
project
  main.py
  requirements.txt            -- Python dependencies
  src/
  |__ config_large_window.py
  |__ model.py
  |__ dataset.py
  |__ visualise.py
  |__ utils.py
  |__ evaluate.py
  |__ train.py
  |__ test_suite_runner.py
  |__ quantitative_metrics.py
  results/                    -- all output CSVs and PNGs
  |__ saved_models
  |__ rest of the plots
```

---

## Run order

```bash
python main.py
```
* Currently ```main.py``` will not retrain the model, to do that set the ```train_again``` flag to ```True``` in ```main.py```. 
---
## Dependencies

```
pandas
scikit-learn
numpy
matplotlib
torch
```

---

## Design decisions

**Why LSTM?**
Very good for keeping track of long sequences.


---
