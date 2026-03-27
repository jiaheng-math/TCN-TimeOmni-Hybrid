from __future__ import annotations


def build_timeomni_question(prediction: dict, summary: dict) -> str:
    sigma = prediction.get("sigma")
    sigma_text = "unavailable" if sigma is None else f"{sigma:.4f}"
    lower = prediction.get("lower_95")
    lower_text = f"{lower:.4f}" if lower is not None else "unavailable"

    trend_block = "\n".join(f"- {line}" for line in summary["trend_lines"])
    return (
        "Assess the maintenance risk for the following engine.\n\n"
        f"Engine ID: {prediction['unit_id']}\n"
        f"Observed cycle: {prediction['observed_cycle']}\n"
        f"TCN model type: {prediction['model_type']}\n"
        f"Predicted RUL: {prediction['predicted_rul']:.4f}\n"
        f"Predictive sigma: {sigma_text}\n"
        f"Lower 95% bound: {lower_text}\n"
        f"Warning level: {prediction['warning']['level']}\n"
        f"Escalated by uncertainty: {prediction['warning']['escalated']}\n"
        f"Summary window: cycles {summary['summary_window']['start_cycle']} to {summary['summary_window']['end_cycle']}\n"
        "Recent feature trends:\n"
        f"{trend_block}\n\n"
        "Explain the risk, recommend a maintenance action, identify the key evidence, "
        "list follow-up checks, and state how much confidence the operator should have."
    )
