# Adversarial Debiasing Experiment

**Paper:** Mitigating Unwanted Biases with Adversarial Learning

**Dataset:** Adult Census Income Dataset  
**Protected attribute:** Gender  
**Metrics:** Demographic parity, Equal opportunity  

**Files:**
- `dataset.py` → loads and preprocesses the Adult dataset
- `model.py` → defines Predictor and network definitions
- `train.py` → training script for adversarial debiasing
- `evaluation.py` → evaluates accuracy and fairness metrics
- `results/` → metrics.csv will be saved
- 'predictor.pt' → saved predictor model (after training)
- 'adversary.pt' → saved adversary model (after training)

## Requirements
- Python 3.8+
-Packages (install with pip):

```bash
pip install torch pandas scikit-learn matplotlib

-'python train.py' → 1. Loads and preprocesses the Adult dataset 2. Trains the Predictor and Adversary networks 3. Prints loss every 10 epochs 4. Saves trained models

-'python evaluation.py' → 1. Loads the trained Predictor 2. Makes predictions on the test set 3. Computes metrics 4. Saves results to results/metrics.csv

