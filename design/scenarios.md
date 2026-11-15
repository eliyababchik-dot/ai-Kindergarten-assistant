# Scenarios

## 1. Filter and rank available substitutes for an absence

The coordinator reports a sudden absence in a kindergarten, and the system filters available substitutes and ranks them by heuristic score (distance, experience, and requirements).

| Absence Request (Kindergarten, Date, Hours) | Available Substitutes Data | Expected Ranked Output (Top Matches) |

| ------------------------------------------- | -------------------------- | ------------------------------------- |

| Garden: Gan Shaked, Date: 2026-11-18, 08:00-14:00 | Sarah (Distance: 1km, Exp: 3 yrs), Leah (Distance: 5km, Exp: 1 yr) | 1. Sarah (Score: 92), 2. Leah (Score: 65) |

| Garden: Gan Rimon, Date: 2026-11-19, 07:30-13:30 | Rachel (Distance: 2km, Exp: 5 yrs), Rivka (Distance: 8km, Exp: 2 yrs) | 1. Rachel (Score: 95), 2. Rivka (Score: 50) |