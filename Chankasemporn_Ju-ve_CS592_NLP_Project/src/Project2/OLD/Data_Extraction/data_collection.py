"""
File Name:    data_collection.py
Author(s):    Ju-ve Chankasemporn
Copyright:    (c) 2025 DigiPen Institute of Technology. All rights reserved.
"""

from collections import Counter
import re

def collectCharacterInfo(root):
    result = []

    cast = root.findall('.//Character')

    for characterInfo in cast:
        full_name = characterInfo.get("name")

        if full_name is None:
            continue

        # Split only on first comma
        parts = full_name.split(",", 1)

        name = parts[0].strip()
        description = parts[1].strip() if len(parts) > 1 else ""

        # Remove trailing period from description
        description = description.rstrip(".")

        result.append({
            "name": name,
            "desc": description
        })

    return result

def collectLocationInfo(node):
    search_root = node

    raw_locations = []

    # Find every Scene that has a location attribute
    for scene in search_root.findall(".//Scene"):
        loc = scene.get("location")
        if not loc:
            continue

        # Normalize whitespace and remove trailing period
        loc = re.sub(r"\s+", " ", loc).strip().rstrip(".")

        if loc:
            raw_locations.append(loc)

    counts = Counter(raw_locations)

    resultList = [
        {"location": loc, "count": cnt}
        for loc, cnt in counts.most_common()
    ]

    return resultList