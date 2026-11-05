PROJECT: 2026 OIBC FINAL

Primary Objective:
Minimize official settlement penalty W.

Rules:
- No external training data.
- No future observation leakage.
- Forecast cutoff: previous day 17:00 KST.
- Use chronological validation.
- Use OOF predictions for stacking.
- Optimize VPP groups using out-of-sample errors.
- Preserve previous best models.
- Never invent benchmark scores.
- Validate submission files before completion.

Repository:
JungGM/OIBC_final
