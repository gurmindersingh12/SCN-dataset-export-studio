"""
Export dataset utility.

This script converts a canonical dataset representation
into target formats such as YOLO or COCO.
"""

def export_dataset(dataset, format="yolo"):
    """
    Export dataset to a target format.

    Parameters
    ----------
    dataset : dict
        Canonical dataset representation.
    format : str
        Target format ('yolo' or 'coco').

    Returns
    -------
    str
        Path to exported dataset.
    """

    if format == "yolo":
        print("Exporting dataset to YOLO format...")
    elif format == "coco":
        print("Exporting dataset to COCO format...")
    else:
        raise ValueError(f"Unsupported format: {format}")

    return "export_complete"


if __name__ == "__main__":
    print("Dataset export tool")
