"""
File Name:    data_extractor.py
Author(s):    Ju-ve Chankasemporn
Copyright:    (c) 2025 DigiPen Institute of Technology. All rights reserved.

Shared utilities for Assignment 2:
  - Repo root discovery
  - XML cleaning and parsing
  - Scene and cast extraction (used by NER, fine-tuning, coreference)
"""

import spacy
import xml.etree.ElementTree as ET
import re
from pathlib import Path

CAST_REL_RE = re.compile(
    r'^(?P<role>.+?)\s+(?P<prep>to|of|on|with|unto)\s+(?P<target>[A-Z][a-zA-Z\s]+?)(?:[,.]|$)',
    re.IGNORECASE
)

def initialize_nlp(model_name: str = "en_core_web_md"):
    return spacy.load(model_name)

# XML PARSING HELPERS  (shared across all modules)
#replace common HTML entities so ET can parse cleanly.
def clean_xml(raw: str) -> str:
    return (raw
        .replace("&#8217;", "'").replace("&#8216;", "'")
        .replace("&#8220;", '"').replace("&#8221;", '"')
        .replace("&#8212;", "—").replace("&#8211;", "–"))

#reada nd parse one play XML file. Returns (xml_root, cleaned_string).
def load_play(filepath: Path):
    raw  = filepath.read_text(encoding="utf-8")
    raw  = clean_xml(raw)
    root = ET.fromstring(raw)
    return root, raw

#extract play title from <Title> tag.
def get_title(root) -> str:
    elem = root.find(".//Title")
    return elem.text.strip() if elem is not None and elem.text else "Unknown"

#return cleaned character names and descriptions from <Character> section.
def extract_cast(root) -> list:
    characters = []
    for char in root.findall(".//Character"):
        raw   = char.attrib.get("name", "")
        parts = raw.split(",", 1)
        name  = parts[0].strip().rstrip(".")
        desc  = parts[1].strip().rstrip(".") if len(parts) > 1 else ""

        if name and not re.match(
            r'^(A |An |The )(Soldier|Porter|Doctor|Man|Boy|Captain|Servant|'
            r'Sexton|Officer|Apothecary|Messenger|Attendant|Musician)', name
        ):
            characters.append({"name": name, "desc": desc})
    return characters

#Parse each character's description to extract relationships using
#prepositions (to, of, on, with, unto) as the relationship connectors.
#Returns list of dicts:
#{ play, source, rel_type, target, description }
def extract_cast_relationships(characters: list, play_title: str) -> list:
    relationships = []

    for char in characters:
        source = char["name"]
        desc = char["desc"]

        if not desc:
            continue

        match = CAST_REL_RE.match(desc.strip())
        if not match:
            continue

        role = match.group("role").strip()  # e.g. "son", "follower"
        prep = match.group("prep").strip()  # e.g. "to", "of"
        target = match.group("target").strip()  # e.g. "Banquo", "Don John"

        rel_type = f"{role}_{prep}".upper().replace(" ", "_")

        record = {
            "play": play_title,
            "source": source,
            "rel_type": rel_type,
            "target": target,
            "description": desc,
        }
        relationships.append(record)
        print(f"    {source}  -[{rel_type}]->  {target}  (from: '{desc}')")

    return relationships

#return list of (act_id, scene_id, location, text) for every scene.
#handles Prologue-style acts (Romeo & Juliet Act 0) that have no
#<Scene> children — text lives directly inside <Act>.
def extract_scenes(root) -> list:
    scenes = []
    for act in root.findall(".//Act"):
        act_id      = act.attrib.get("id", "?")
        scene_elems = act.findall("Scene")
        if scene_elems:
            for scene in scene_elems:
                scene_id = scene.attrib.get("id", "?")
                location = scene.attrib.get("location", "")
                text     = re.sub(r'\s+', ' ', " ".join(scene.itertext()).strip())
                if text:
                    scenes.append((act_id, scene_id, location, text))
        else:
            text = re.sub(r'\s+', ' ', " ".join(act.itertext()).strip())
            if text:
                scenes.append((act_id, "0", "Prologue", text))
    return scenes