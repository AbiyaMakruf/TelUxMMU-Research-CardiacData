class Config:
    """Configuration class for ECG preprocessing pipeline.

    Contains all constants and parameters used throughout the preprocessing workflow,
    including dataset URLs, folder mappings, and coordinate definitions for image cropping.
    """
    seed: int = 42
    data_url: str = "https://data.mendeley.com/public-api/zip/gwbz3fsgp8/download/2"
    zip_name: str = "gwbz3fsgp8-2.zip"
    data_root: str = "data"
    folder_mapping: dict = {
        "ECG Images of Myocardial Infarction Patients (240x12=2880)": "ECG_MI",
        "ECG Images of Patient that have abnormal heartbeat (233x12=2796)": "ECG_Abnormal",
        "ECG Images of Patient that have History of MI (172x12=2064)": "ECG_HistoryMI",
        "Normal Person ECG Images (284x12=3408)": "ECG_Normal"
    } 
    ecg_area_coordinates: dict = {
        "left":130,
        "top":290,
        "right":2170,
        "bottom": 1505
    }

    short_lead_coordinates: dict = {
        "x1": 40,
        "y1": 40,
        "x2": 4000,
        "y2": 1800
    }
    
    long_lead_coordinates: dict = {
        "x1": 40,
        "y1": 2000,
        "x2": 4000,
        "y2": 2300
    }

    lead_mapping: dict = {
        1: "lead_I",   2: "a_VR",  3: "v_1",  4: "v_4",
        5: "lead_II",  6: "a_VL",  7: "v_2",  8: "v5",
        9: "lead_III", 10: "a_VF", 11: "v_3", 12: "v_6"
    }