"""Teacher v2 — retrospective label generation for the Student brain."""
from teacher.teacher_v2 import (
    compute_teacher_labels,
    TeacherLabels,
    SL_FLOOR, SL_CAP,
    TP_FLOOR, TP_CAP,
    SIZE_FLOOR, SIZE_CAP,
    ENTRY_MFE_MIN_R, ENTRY_MFE_OVER_MAE_RATIO,
)

__all__ = [
    "compute_teacher_labels",
    "TeacherLabels",
    "SL_FLOOR", "SL_CAP",
    "TP_FLOOR", "TP_CAP",
    "SIZE_FLOOR", "SIZE_CAP",
    "ENTRY_MFE_MIN_R", "ENTRY_MFE_OVER_MAE_RATIO",
]
