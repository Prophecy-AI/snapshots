import json
import os
from datetime import datetime

# Read session state
with open('session_state.json', 'r') as f:
    state = json.load(f)

# Create new experiment
exp_id = f"exp_{len(state['experiments']):03d}"
experiment = {
    "id": exp_id,
    "name": "005_strict_ensemble",
    "model_type": "ensemble",
    "score": 70.615745,
    "cv_score": 70.615745,
    "lb_score": None,
    "notes": "Strict ensemble picks best VALID config per N from 30+ CSV sources. Validated with Shapely (threshold 1e-15) - no overlaps. Score: 70.615745 vs baseline 70.676102 (improvement: 0.060357). Improves 162 out of 200 N values. This is the best score achieved so far.",
    "experiment_folder": "experiments/005_ensemble",
    "timestamp": datetime.now().isoformat()
}

state['experiments'].append(experiment)

# Add candidate
candidate = {
    "file_path": "/home/code/submission_candidates/candidate_004.csv",
    "score": 70.615745,
    "cv_score": 70.615745,
    "model_name": "005_strict_ensemble",
    "experiment_id": exp_id,
    "timestamp": datetime.now().isoformat()
}
state['candidates'].append(candidate)

# Save
with open('session_state.json', 'w') as f:
    json.dump(state, f, indent=2)

# Copy submission
os.makedirs('/home/code/submission_candidates', exist_ok=True)
import shutil
shutil.copy('experiments/005_ensemble/submission_ensemble_strict.csv', 
            '/home/code/submission_candidates/candidate_004.csv')

print(f"Logged experiment {exp_id}")
print(f"Score: 70.615745")
