import joblib
from pathlib import Path

p = Path("artifacts/lgbm_model.pkl")
obj = joblib.load(p)
print("Loaded:", type(obj))

# If it's a sklearn Pipeline, it will have 'steps' and predict on raw df will work.
has_steps = hasattr(obj, "steps")
print("Has .steps:", has_steps)

if has_steps:
    print("Pipeline steps:", [name for name, _ in obj.steps])
