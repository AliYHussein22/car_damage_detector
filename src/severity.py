"""
src/severity.py
---------------
Turns a list of detections into a single severity score and label.

The scoring is intentionally simple — each detection contributes a weighted
confidence value, and we normalise the total to a 0-10 scale. The thresholds
(3.5 / 6.5) were tuned so that a single confident scratch reads as Low,
a couple of dents reads as Medium, and anything with broken parts or shattered
glass lands in High.
"""

from __future__ import annotations

# how much each damage type contributes to the overall score
# heavier damage types have higher weights so they push the score up faster
DAMAGE_WEIGHTS = {
    "broken-part":      3.0,
    "shattered-glass":  2.5,
    "crack":            2.0,
    "flat-tyre":        2.0,
    "dent":             1.5,
    "scratch":          1.0,
}
DEFAULT_WEIGHT = 1.5  # fallback for any class not in the table above


def compute_severity(detections: list[dict]) -> dict:
    """
    Compute an overall damage severity score from a list of detections.

    Parameters
    ----------
    detections : list of dicts, each with keys 'label' and 'confidence'

    Returns
    -------
    dict with:
        score : float  — damage score on a 0-10 scale
        level : str    — "Low", "Medium", or "High" (or "None" if nothing detected)
    """
    if not detections:
        return {"score": 0.0, "level": "None"}

    # sum weighted confidences across all detections
    raw = sum(
        DAMAGE_WEIGHTS.get(d["label"], DEFAULT_WEIGHT) * d["confidence"]
        for d in detections
    )

    # normalise so that roughly 5 high-confidence serious detections = 10/10
    score = min(raw / 5.0 * 10, 10.0)

    if score < 3.5:
        level = "Low"
    elif score < 6.5:
        level = "Medium"
    else:
        level = "High"

    return {"score": round(score, 2), "level": level}
