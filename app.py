import gradio as gr
from functools import partial
from pathlib import Path

from services.prediction import (
    MODEL_METADATA,
    PERCENTAGE_FEATURES,
    classifier_global_drivers,
    predict_site,
)

ROOT = Path(__file__).resolve().parent

THEME = gr.themes.Base(
    primary_hue="teal",
    secondary_hue="slate",
    neutral_hue="gray",
    spacing_size="sm",
    radius_size="sm",
    text_size="md",
).set(
    slider_color="#0f766e",
    body_text_color_subdued="#536963",
    block_info_text_color="#536963",
)


LAND_COVER_INPUTS = [
    ("pct_cropping_horticulture", "Cropping and horticulture", 1.5),
    ("pct_indigenous_forest", "Indigenous forest", 21.0),
    ("pct_exotic_forest", "Exotic forest", 9.3),
    ("pct_exotic_grassland", "Exotic grassland", 46.4),
    ("pct_tussock_grassland", "Tussock grassland", 6.3),
    (
        "pct_other_herbaceous_vegetation",
        "Other herbaceous vegetation",
        0.5,
    ),
    (
        "pct_indigenous_scrub_shrubland",
        "Indigenous scrub and shrubland",
        5.6,
    ),
    (
        "pct_exotic_scrub_shrubland",
        "Exotic scrub and shrubland",
        1.2,
    ),
    ("pct_urban_area", "Urban area", 4.4),
    (
        "pct_natural_bare_lightly-vegetated_surfaces",
        "Natural bare or lightly vegetated surfaces",
        1.9,
    ),
    (
        "pct_artificial_bare_surfaces",
        "Artificial bare surfaces",
        0.1,
    ),
]

INPUT_ORDER = [
    item[0] for item in LAND_COVER_INPUTS
] + [
    "mean_annual_rainfall_mm",
    "wfs_altitude",
    "rec_land_cover_type",
]

PRESETS = {
    "Lowland pasture": (
        0.99, 16.08, 16.60, 53.40, 0.55, 0.62, 3.39, 0.48, 1.40, 0.69,
        0.20, 1373, "lowland", "Pasture",
    ),
    "Urban lowland": (
        0.30, 3.58, 1.39, 21.64, 0.00, 0.00, 0.18, 0.00, 72.48, 0.00,
        0.35, 1370, "lowland", "Urban",
    ),
    "Upland native vegetation": (
        0.22, 33.12, 4.70, 56.13, 0.10, 0.05, 4.27, 0.14, 0.40, 0.28,
        0.04, 2029, "upland", "Native vegetation",
    ),
}

BAND_INTERPRETATIONS = {
    "A": (
        "The model indicates a comparatively low contamination risk. "
        "Routine monitoring should nevertheless continue."
    ),
    "B": (
        "This is an intermediate band for which model discrimination is "
        "limited. Confirmatory field sampling is recommended."
    ),
    "C": (
        "This is an intermediate band for which model discrimination is "
        "limited. Management decisions should not rely on this prediction alone."
    ),
    "D": (
        "The result indicates a potentially elevated contamination risk. "
        "The site should be prioritised for confirmatory sampling."
    ),
    "E": (
        "The result indicates potentially severe contamination. Prompt site "
        "investigation and confirmatory microbiological sampling are recommended."
    ),
}

BAND_COLORS = {
    "A": "#2f855a",
    "B": "#65a30d",
    "C": "#d4a72c",
    "D": "#ea7c2b",
    "E": "#c2413b",
}

MODEL_LIMITATION = (
    "This prototype provides catchment-scale screening support only. It is not "
    "a regulatory assessment and does not replace microbiological sampling, "
    "site-specific evidence, or professional judgement."
)


def land_cover_status_html(*percentages):
    total = sum(float(value) for value in percentages)
    if total > 100.5:
        state = "invalid"
        message = "exceeds 100% — revise inputs"
    elif total < 90.3:
        state = "caution"
        message = "below the observed 90.3–100.0% range"
    else:
        state = "valid"
        message = "within the observed range"

    return f"""
    <div class="cover-total cover-total--{state}">
        <strong>Total {total:.1f}%</strong><span>{message}</span>
    </div>
    """


def preset_assessment_output(name):
    """Apply a representative profile and return its assessment atomically."""
    values = PRESETS[name]
    summary, score_chart, interpretation = run_prediction(*values)
    return (
        *values,
        land_cover_status_html(*values[: len(PERCENTAGE_FEATURES)]),
        summary,
        score_chart,
        interpretation,
    )


def reset_land_cover_inputs():
    defaults = tuple(default for _, _, default in LAND_COVER_INPUTS)
    return (*defaults, land_cover_status_html(*defaults))


def band_score_chart_html(scores=None, winner=None):
    scores = scores or {band: 0.0 for band in "ABCDE"}
    rows = []
    for band in "ABCDE":
        share = float(scores.get(band, 0.0))
        width = max(0.0, min(100.0, share * 100))
        winner_class = " band-bar--winner" if band == winner else ""
        winner_text = " · predicted" if band == winner else ""
        rows.append(
            f"""
            <div class="band-bar{winner_class}" style="--band-color: {BAND_COLORS[band]}">
                <div class="band-bar__label">Band {band}{winner_text}</div>
                <div class="band-bar__track">
                    <span class="band-bar__fill" style="width: {width:.1f}%"></span>
                </div>
                <div class="band-bar__value">{share:.0%}</div>
            </div>
            """
        )
    return '<div class="band-chart">' + "".join(rows) + "</div>"


def reset_result():
    return (
        result_summary_html(),
        band_score_chart_html(),
        "Inputs changed. Run the assessment to generate an updated result.",
    )


def result_summary_html(result=None):
    if result is None:
        median_text = "—"
        interval_text = "Empirical interval shown after assessment"
        band_text = "—"
        vote_text = "Tree vote share shown after assessment"
        band_class = "band-none"
    else:
        median_text = f"{result['predicted_median']:.1f}"
        interval = result["regression_interval"]
        interval_text = (
            f"Approx. 80% empirical range: {interval['lower']:.0f}–"
            f"{interval['upper']:.0f} MPN/100 mL"
        )
        band_text = f"Band {result['predicted_band']}"
        vote_text = (
            f"{result['predicted_band_vote_share']:.0%} of trees voted for "
            f"Band {result['predicted_band']}"
        )
        band_class = f"band-{result['predicted_band'].lower()}"

    return f"""
    <section class="result-summary" aria-label="Screening result summary">
        <article class="result-metric">
            <div class="result-metric__label">Estimated E. coli median</div>
            <div class="result-metric__value">{median_text}</div>
            <span class="result-metric__unit">MPN / 100 mL</span>
            <span class="result-metric__context">{interval_text}</span>
            <span class="result-metric__context">
                Held-out R² = {MODEL_METADATA['regression_test_r2']:.2f}; catchment
                features explain only part of the observed variation.
            </span>
        </article>
        <article class="result-metric result-metric--band {band_class}">
            <div class="result-metric__label">NPS-FM attribute band</div>
            <div class="result-metric__value band-value">{band_text}</div>
            <span class="result-metric__unit">{vote_text}</span>
        </article>
    </section>
    """


def run_prediction(*args):
    values = dict(zip(INPUT_ORDER, args, strict=True))

    try:
        result = predict_site(values)
    except (TypeError, ValueError) as error:
        raise gr.Error(str(error)) from error

    predicted_band = result["predicted_band"]

    score_chart = band_score_chart_html(result["band_scores"], predicted_band)

    reliability_warning = ""
    if result["reliability_notes"]:
        reliability_warning = (
            "> **Lower-reliability profile.** "
            + " ".join(result["reliability_notes"])
            + "\n\n"
        )

    interpretation = (
        "### Screening interpretation\n\n"
        f"{reliability_warning}"
        f"{BAND_INTERPRETATIONS[predicted_band]}\n\n"
        "The concentration estimate and attribute band come from two independently "
        "trained models with different targets. The band is not calculated from the "
        "displayed median, so the two outputs may not align exactly."
    )

    return (
        result_summary_html(result),
        score_chart,
        interpretation,
    )


def percentage_slider(label, value):
    return gr.Slider(
        minimum=0,
        maximum=100,
        value=value,
        step=0.1,
        label=label,
        elem_classes=["land-cover-slider"],
    )


GLOBAL_DRIVER_ROWS = [
    [driver["feature"], round(driver["importance"], 3)]
    for driver in classifier_global_drivers()
]


with gr.Blocks(title="AwaScreen | E. coli Screening") as demo:
    gr.HTML(
        """
        <header class="hero">
            <h1>AwaScreen</h1>
            <p class="hero__description">
                Catchment-scale E. coli screening for New Zealand rivers.
            </p>
        </header>
        """,
        elem_classes=["hero-host"],
    )

    with gr.Row(elem_classes=["dashboard-grid"]):
        with gr.Column(scale=3, elem_classes=["panel", "input-panel"]):
            gr.Markdown("SITE INPUTS", elem_classes=["section-kicker"])
            gr.Markdown("## Catchment profile", elem_classes=["section-heading"])

            with gr.Row():
                altitude = gr.Dropdown(
                    choices=[("Lowland", "lowland"), ("Upland", "upland")],
                    value="lowland",
                    label="Altitude class",
                )

                land_cover_type = gr.Dropdown(
                    choices=[
                        "Exotic forest",
                        "Native vegetation",
                        "Pasture",
                        "Urban",
                    ],
                    value="Pasture",
                    label="Dominant land-cover class",
                )

            rainfall = gr.Slider(
                minimum=542,
                maximum=3373,
                value=1487,
                step=1,
                label="Mean annual rainfall",
                info="Millimetres per year · constrained to the observed training range",
            )

            gr.Markdown("Representative profiles", elem_classes=["preset-label"])
            preset_buttons = {}
            with gr.Row(elem_classes=["preset-row"]):
                for preset_name in PRESETS:
                    preset_buttons[preset_name] = gr.Button(
                        preset_name,
                        size="sm",
                        variant="secondary",
                    )

            percentage_components = {}

            with gr.Row(elem_classes=["composition-heading"]):
                gr.Markdown("**Land-cover composition**")
                cover_total = gr.HTML(
                    land_cover_status_html(
                        *(default for _, _, default in LAND_COVER_INPUTS)
                    )
                )

            with gr.Accordion(
                "Adjust percentage inputs",
                open=False,
            ):
                gr.Markdown(
                    "Each value is the percentage of the upstream catchment.",
                    elem_classes=["composition-help"],
                )
                with gr.Row():
                    with gr.Column():
                        for key, label, default in LAND_COVER_INPUTS[:6]:
                            percentage_components[key] = percentage_slider(
                                label,
                                default,
                            )

                    with gr.Column():
                        for key, label, default in LAND_COVER_INPUTS[6:]:
                            percentage_components[key] = percentage_slider(
                                label,
                                default,
                            )

            percentage_inputs = [
                percentage_components[key]
                for key, _, _ in LAND_COVER_INPUTS
            ]
            reset_cover_button = gr.Button(
                "Reset land-cover inputs",
                size="sm",
                variant="secondary",
                elem_classes=["reset-cover"],
            )

            run_button = gr.Button(
                "Run screening assessment",
                variant="primary",
                elem_classes=["primary-action"],
            )

        with gr.Column(scale=2, elem_classes=["panel", "results-panel"]):
            gr.Markdown("MODEL OUTPUT", elem_classes=["section-kicker"])
            gr.Markdown("## Screening result", elem_classes=["section-heading"])

            result_summary = gr.HTML(result_summary_html())

            interpretation = gr.Markdown(
                "Run the assessment to generate an interpretation.",
                elem_classes=["interpretation-card"],
            )

            gr.Markdown(
                "**Geographic coverage:** Canterbury is absent from the training "
                "dataset. Predictions for Canterbury-like catchments should be "
                "treated as preliminary until locally validated.",
                elem_classes=["coverage-note"],
            )

            with gr.Accordion("Classification diagnostics", open=False):
                gr.Markdown("**Random-forest vote distribution**")
                band_scores = gr.HTML(band_score_chart_html())

                gr.Markdown(
                    "Vote shares are not calibrated probabilities. Model performance "
                    "is strongest at the clean and severely contaminated ends of the "
                    "classification range.",
                    elem_classes=["model-note"],
                )

                gr.Dataframe(
                    value=GLOBAL_DRIVER_ROWS,
                    headers=["Feature", "Global importance"],
                    datatype=["str", "number"],
                    interactive=False,
                    label="Top global classifier drivers",
                )

                gr.Markdown(
                    "These are global random-forest importances, not causal effects "
                    "or a local explanation of this individual prediction. Local SHAP "
                    "values are intentionally deferred until the encoded feature "
                    "pipeline has been separately validated.",
                    elem_classes=["model-note"],
                )

    gr.Markdown(
        f"**Use limitation.** {MODEL_LIMITATION}",
        elem_classes=["scope-note"],
    )

    gr.HTML(
        f"""
        <footer class="project-footer">
            <div>
                <strong>Models</strong><br>
                MLP regression + balanced random-forest classification
            </div>
            <div>
                <strong>Evaluation sample</strong><br>
                {MODEL_METADATA['labelled_sites']} labelled sites ·
                {MODEL_METADATA['training_sites']} train ·
                {MODEL_METADATA['test_sites']} test
            </div>
            <div>
                <strong>Data and version</strong><br>
                Project dataset with NPS-FM 2020 labels · 4 October 2026
            </div>
            <a href="https://github.com/Ming-Liu-1170030/AwaScreen"
               target="_blank" rel="noopener noreferrer">View source on GitHub ↗</a>
        </footer>
        """
    )

    model_inputs = [
        *percentage_inputs,
    ] + [
        rainfall,
        altitude,
        land_cover_type,
    ]

    run_button.click(
        fn=run_prediction,
        inputs=model_inputs,
        outputs=[
            result_summary,
            band_scores,
            interpretation,
        ],
    )

    for component in percentage_inputs:
        component.input(
            fn=land_cover_status_html,
            inputs=percentage_inputs,
            outputs=cover_total,
        )

    for preset_name, button in preset_buttons.items():
        button.click(
            fn=partial(preset_assessment_output, preset_name),
            inputs=None,
            outputs=[
                *model_inputs,
                cover_total,
                result_summary,
                band_scores,
                interpretation,
            ],
        )

    reset_cover_button.click(
        fn=reset_land_cover_inputs,
        inputs=None,
        outputs=[*percentage_inputs, cover_total],
    )

    for component in model_inputs:
        component.input(
            fn=reset_result,
            inputs=None,
            outputs=[result_summary, band_scores, interpretation],
        )


if __name__ == "__main__":
    demo.launch(
        theme=THEME,
        css_paths=ROOT / "assets" / "style.css",
    )
