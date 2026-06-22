LIMB_LEADS = [
    "01_lead_lead_I",
    "05_lead_lead_II",
    "09_lead_lead_III",
    "02_lead_a_VR",
    "06_lead_a_VL",
    "10_lead_a_VF",
]

PRECORDIAL_LEADS = [
    "03_lead_v_1",
    "07_lead_v_2",
    "11_lead_v_3",
    "04_lead_v_4",
    "08_lead_v5",
    "12_lead_v_6",
]

LONG_LEAD = "13_long_lead"
SHORT_12_LEADS = LIMB_LEADS + PRECORDIAL_LEADS
GRID_12_LEADS = [
    "01_lead_lead_I",
    "02_lead_a_VR",
    "03_lead_v_1",
    "04_lead_v_4",
    "05_lead_lead_II",
    "06_lead_a_VL",
    "07_lead_v_2",
    "08_lead_v5",
    "09_lead_lead_III",
    "10_lead_a_VF",
    "11_lead_v_3",
    "12_lead_v_6",
]
ALL_13_LEADS = SHORT_12_LEADS + [LONG_LEAD]
GRID_13_LEADS = GRID_12_LEADS + [LONG_LEAD]


def lead_column_name(prefix: str) -> str:
    number = prefix.split("_", 1)[0]
    return f"lead_{number}_path"
