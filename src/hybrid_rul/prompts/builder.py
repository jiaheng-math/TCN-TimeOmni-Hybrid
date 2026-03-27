from __future__ import annotations


# Mapping from warning level to recommended action window.
_ACTION_WINDOWS: dict[str, str] = {
    "正常": "routine monitoring",
    "关注": "next 20 cycles",
    "预警": "next 10 cycles",
    "危险": "immediate",
}


def build_timeomni_question(prediction: dict, summary: dict, thresholds: dict | None = None) -> str:
    sigma = prediction.get("sigma")
    sigma_text = "unavailable" if sigma is None else f"{sigma:.4f}"
    lower = prediction.get("lower_95")
    lower_text = f"{lower:.4f}" if lower is not None else "unavailable"
    thresholds = thresholds or {"normal": 80, "watch": 50, "alert": 20}

    warning = prediction["warning"]
    level = warning["level"]
    escalated = warning.get("escalated", False)
    action_window = _ACTION_WINDOWS.get(level, "routine monitoring")

    # Build escalation explanation when applicable.
    if escalated:
        escalation_note = (
            "Note: The warning level has been escalated one tier above the raw "
            "threshold result because the predictive uncertainty (sigma) is high. "
            "This is a deliberate safety policy. Use the given warning level, not "
            "the raw thresholds, for your recommendation."
        )
    else:
        escalation_note = (
            "Note: The warning level was determined directly from the thresholds "
            "without any uncertainty escalation."
        )

    trend_block = "\n".join(f"- {line}" for line in summary["trend_lines"])
    return (
        "Assess the maintenance risk for the following engine.\n\n"
        f"Engine ID: {prediction['unit_id']}\n"
        f"Observed cycle: {prediction['observed_cycle']}\n"
        f"Model backbone: {prediction.get('model_backbone', 'tcn')}\n"
        f"Model type: {prediction['model_type']}\n"
        f"Predicted RUL: {prediction['predicted_rul']:.4f}\n"
        f"Predictive sigma: {sigma_text}\n"
        f"Lower 95% bound: {lower_text}\n"
        f"Exact warning level token: {level}\n"
        f"Escalated by uncertainty: {escalated}\n"
        f"{escalation_note}\n"
        f"Recommended action for this warning level: {action_window}\n"
        f"Summary window: cycles {summary['summary_window']['start_cycle']} "
        f"to {summary['summary_window']['end_cycle']}\n"
        "Recent feature trends:\n"
        f"{trend_block}\n\n"
        "Based on the above, produce your assessment in the five required tags. "
        "Use the given warning level and its corresponding action window as your starting point."
    )
