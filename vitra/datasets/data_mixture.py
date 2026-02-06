from typing import Dict, List, Tuple

HAND_MIXTURES: Dict[str, List[Tuple[str, float]]] = {
    # === ego4d + egoexo4d + ssv2 + epic Dataset ===
    "magic_mix": [
        ("ego4d_cooking_and_cleaning", 1.0),
        ("egoexo4d", 3.0),
        ("epic", 1.0),
        ('ssv2', 5.0),
        ('ego4d_other', 0.5)
    ],
    "magic_mix_cooking_and_cleaning": [
        ("ego4d_cooking_and_cleaning", 1.0),
        ("egoexo4d", 3.0),
        ("epic", 1.0),
        ('ssv2', 5.0),
    ],
    "ssv2": [
        ('ssv2', 1.0)
    ],
}
