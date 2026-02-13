"""
Maps raw YOLO classes into domain-specific meanings.
This lets us adapt pretrained models without retraining.
"""

# Classes YOLO *might* use to describe a carried object
CARRY_ITEM_ALIASES = {
    "handbag",
    "backpack",
    "suitcase",
    "briefcase",
    "tie",          # often misclassified purse strap
    "book",         # small pouch sometimes becomes book
    "cell phone",   # tiny clutch gets tagged like this
}


def map_to_domain(label: str):
    """
    Convert YOLO label into system-level meaning.
    """
    if label in CARRY_ITEM_ALIASES:
        return "personal_item"

    return None
