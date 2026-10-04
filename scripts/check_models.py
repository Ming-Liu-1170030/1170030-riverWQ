from pathlib import Path

import pandas as pd

from services.prediction import FEATURE_COLUMNS, predict_site


ROOT = Path(__file__).resolve().parents[1]

data = pd.read_csv(ROOT / "data" / "riverWQ_dataset.csv")
valid_data = data.dropna(subset=["ecoli_median", "ecoli_band"])

sample = valid_data.iloc[0]
features = sample[FEATURE_COLUMNS].to_dict()
result = predict_site(features)

print("Site:", sample["site_id"])
print("Actual median:", sample["ecoli_median"])
print("Predicted median:", round(result["predicted_median"], 1))
print("Actual band:", sample["ecoli_band"])
print("Predicted band:", result["predicted_band"])
print(
    "Band scores:",
    {
        band: round(score, 3)
        for band, score in result["band_scores"].items()
    },
)