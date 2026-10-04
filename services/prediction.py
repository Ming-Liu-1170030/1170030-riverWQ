from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

FEATURE_COLUMNS = [
    "pct_cropping_horticulture",
    "pct_indigenous_forest",
    "pct_exotic_forest",
    "pct_exotic_grassland",
    "pct_tussock_grassland",
    "pct_other_herbaceous_vegetation",
    "pct_indigenous_scrub_shrubland",
    "pct_exotic_scrub_shrubland",
    "pct_urban_area",
    "pct_natural_bare_lightly-vegetated_surfaces",
    "pct_artificial_bare_surfaces",
    "mean_annual_rainfall_mm",
    "wfs_altitude",
    "rec_land_cover_type",
]

PERCENTAGE_FEATURES = FEATURE_COLUMNS[:11]

ALTITUDE_CHOICES = {"lowland", "upland"}

LAND_COVER_CHOICES = {
    "Exotic forest",
    "Native vegetation",
    "Pasture",
    "Urban",
}

# Reproduced from the notebook's stratified 80/20 split (random_state=42).
# These are the 10th and 90th percentiles of actual-minus-predicted residuals
# on the 68-site held-out set. They form a descriptive, not calibrated,
# empirical 80% interval around a new point estimate.
REGRESSION_RESIDUAL_INTERVAL = (-97.19824016552728, 523.6047289919139)

MODEL_METADATA = {
    "labelled_sites": 339,
    "training_sites": 271,
    "test_sites": 68,
    "regression_test_r2": 0.2424492416813524,
    "classification_test_accuracy": 0.6323529411764706,
    "classification_test_macro_f1": 0.52582145901839,
}

FEATURE_LABELS = {
    "mean_annual_rainfall_mm": "Mean annual rainfall",
    "pct_natural_bare_lightly-vegetated_surfaces": (
        "Natural bare or lightly vegetated surfaces"
    ),
    "pct_tussock_grassland": "Tussock grassland",
    "pct_other_herbaceous_vegetation": "Other herbaceous vegetation",
    "pct_indigenous_scrub_shrubland": "Indigenous scrub and shrubland",
    "pct_urban_area": "Urban area",
    "pct_indigenous_forest": "Indigenous forest",
}


@lru_cache(maxsize=1)
def load_models():
    """Load the two trusted local model files once."""

    regression_model = joblib.load(
        ROOT / "models" / "1_rg_mdl_regression_mlp.pkl"
    )
    classification_model = joblib.load(
        ROOT / "models" / "2_cf_mdl_classification_random_forest.pkl"
    )

    return regression_model, classification_model


def prepare_input(values: dict[str, Any]) -> pd.DataFrame:
    """Validate one site's features and create the model input."""

    missing = set(FEATURE_COLUMNS) - set(values)
    if missing:
        raise ValueError(f"Missing features: {sorted(missing)}")

    record = {}

    for feature in PERCENTAGE_FEATURES:
        value = float(values[feature])

        if not 0 <= value <= 100:
            raise ValueError(f"{feature} must be between 0 and 100")

        record[feature] = value

    percentage_total = sum(record.values())
    if percentage_total > 100.5:
        raise ValueError(
            f"Land-cover percentages total {percentage_total:.1f}%"
        )

    rainfall = float(values["mean_annual_rainfall_mm"])
    if not 0 < rainfall <= 10_000:
        raise ValueError(
            "mean_annual_rainfall_mm must be between 0 and 10,000"
        )

    record["mean_annual_rainfall_mm"] = rainfall

    altitude = values["wfs_altitude"]
    if altitude not in ALTITUDE_CHOICES:
        raise ValueError(f"Unsupported altitude: {altitude}")

    land_cover = values["rec_land_cover_type"]
    if land_cover not in LAND_COVER_CHOICES:
        raise ValueError(f"Unsupported land-cover type: {land_cover}")

    record["wfs_altitude"] = altitude
    record["rec_land_cover_type"] = land_cover

    return pd.DataFrame([record], columns=FEATURE_COLUMNS)


def regression_interval(point_estimate: float) -> tuple[float, float]:
    """Apply held-out residual quantiles to a regression point estimate."""

    lower_offset, upper_offset = REGRESSION_RESIDUAL_INTERVAL
    return (
        max(0.0, point_estimate + lower_offset),
        max(0.0, point_estimate + upper_offset),
    )


def _display_feature_name(transformed_name: str) -> str:
    name = transformed_name.removeprefix("num__").removeprefix("cat__")
    if name == "wfs_altitude_upland":
        return "Upland altitude class"
    if name.startswith("rec_land_cover_type_"):
        category = name.removeprefix("rec_land_cover_type_")
        return f"Dominant land cover: {category}"
    return FEATURE_LABELS.get(name, name.replace("_", " ").title())


@lru_cache(maxsize=1)
def classifier_global_drivers(limit: int = 3) -> tuple[dict[str, Any], ...]:
    """Return global RF feature importances; these are not local SHAP values."""

    _, classification_model = load_models()
    preprocessor = classification_model.named_steps["pre"]
    estimator = classification_model.named_steps["model"]
    names = preprocessor.get_feature_names_out()
    ranked = sorted(
        zip(names, estimator.feature_importances_, strict=True),
        key=lambda item: item[1],
        reverse=True,
    )[:limit]
    return tuple(
        {
            "feature": _display_feature_name(str(name)),
            "importance": float(importance),
        }
        for name, importance in ranked
    )


def predict_site(values: dict[str, Any]) -> dict[str, Any]:
    """Run the regression and classification models for one site."""

    model_input = prepare_input(values)
    regression_model, classification_model = load_models()

    predicted_median = max(
        0.0,
        float(regression_model.predict(model_input)[0]),
    )
    predicted_band = str(
        classification_model.predict(model_input)[0]
    )

    raw_scores = classification_model.predict_proba(model_input)[0]
    band_scores = {
        str(label): float(score)
        for label, score in zip(
            classification_model.classes_,
            raw_scores,
            strict=True,
        )
    }

    percentage_total = sum(float(values[name]) for name in PERCENTAGE_FEATURES)
    interval_lower, interval_upper = regression_interval(predicted_median)
    reliability_notes = []
    if (
        values["wfs_altitude"] == "lowland"
        and values["rec_land_cover_type"] in {"Pasture", "Urban"}
    ):
        reliability_notes.append(
            "Model reliability was lower for lowland Pasture and Urban sites "
            "during subgroup evaluation. Treat this result as a prioritisation "
            "signal and confirm it with field sampling."
        )

    return {
        "predicted_median": predicted_median,
        "regression_interval": {
            "coverage": 0.80,
            "lower": interval_lower,
            "upper": interval_upper,
        },
        "predicted_band": predicted_band,
        "predicted_band_vote_share": band_scores[predicted_band],
        "band_scores": band_scores,
        "land_cover_total": percentage_total,
        "reliability_notes": reliability_notes,
    }
