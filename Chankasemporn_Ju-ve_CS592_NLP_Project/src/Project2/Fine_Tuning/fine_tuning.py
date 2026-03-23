"""
File Name:    fine_tuning.py
Author(s):    Ju-ve Chankasemporn
Copyright:    (c) 2025 DigiPen Institute of Technology. All rights reserved.

Build order
-----------
Training data is assembled in two layers, in strict order:

  Layer 1 — Characters + Relationships  (authoritative, built first)
      Build examples from the character list (play_configs + XML cast)
      and cast relationships.  Every name registered here is added to a
      "known entity" set.  These examples are the ground truth the model
      must learn.

  Layer 2 — Default NER CSV  (supplementary, filtered)
      Load the default NER CSV from Step 1.  For each entity:
        a) Correct its label via play_configs (same priority as before).
        b) OVERLAP CHECK — if the entity text contains, or is contained
           by, any name already registered in Layer 1, SKIP it entirely.
           This prevents composite noise like "MACBETH AND BANQUO",
           "WORTHY MACBETH", "MACBETH." or "WORK_OF_ART: MACBETH"
           from polluting the training data.
        c) Otherwise add it as a supplementary example (GPE, LOC, DATE…).

  Then: balance → fine-tune → extract & save.
"""

import spacy
from spacy.training import Example
from spacy.util import minibatch, compounding
import re
import random
import csv
from pathlib import Path
from collections import defaultdict

import src.Project2.Project2Globals as Globals
import src.Project2.Data_Extraction.data_extractor as DataExtractor
from src.Project2.Play_Configs.play_configs import PLAY_CONFIGS

# ── helpers ──────────────────────────────────────────────────────────────────

def find_all_spans(text: str, phrase: str) -> list:
    """Find all non-overlapping occurrences of phrase in text (case-insensitive)."""
    spans      = []
    start      = 0
    text_lower = text.lower()
    phr_lower  = phrase.lower()
    while True:
        idx = text_lower.find(phr_lower, start)
        if idx == -1:
            break
        spans.append((idx, idx + len(phrase)))
        start = idx + len(phrase)
    return spans

def remove_overlaps(entities: list) -> list:
    """Keep first entity when spans overlap."""
    result, last_end = [], -1
    for ent in sorted(entities, key=lambda x: x[0]):
        if ent[0] >= last_end:
            result.append(ent)
            last_end = ent[1]
    return result


# ── shared: resolve a corrected label from config ────────────────────────────

def _resolve_label(entity_text: str, spacy_label: str,
                   person_set: set, gpe_set: set,
                   loc_set: set, title_set: set,
                   char_whitelist: set):
    """
    Return corrected label for one entity, or None if it should be dropped.

    Priority:
      1. In person_set                  → PERSON
      2. In gpe_set                     → GPE
      3. In loc_set                     → LOCATION
      4. In title_set                   → TITLE
      5. spaCy said PERSON but unknown  → None  (drop noise)
      6. Everything else                → keep spaCy label unchanged
    """
    upper = entity_text.upper()
    if upper in person_set:  return "PERSON"
    if upper in gpe_set:     return "GPE"
    if upper in loc_set:     return "LOCATION"
    if upper in title_set:   return "TITLE"
    if spacy_label == "PERSON" and upper not in char_whitelist:
        return None
    return spacy_label


def _overlaps_with_known(entity_text: str, known_names: set) -> bool:
    """
    Return True if entity_text overlaps with any already-registered name.

    Two kinds of overlap:
      - Substring:  "MACBETH AND BANQUO" contains "MACBETH"  → True
      - Superstring: a known name contains the entity text   → True
      - Exact match is also caught by substring check
    """
    upper = entity_text.upper()
    for name in known_names:
        if name in upper or upper in name:
            return True
    return False


# ── LAYER 1: characters + relationships ──────────────────────────────────────

def build_layer1_examples(characters: list, cast_relationships: list,
                           char_whitelist: set, config: dict,
                           nlp, play_title: str) -> tuple:
    """
    Layer 1 — build authoritative training examples from:
      a) Character list: one "X is a character" / "X, <desc>" sentence each
      b) Cast relationships: four synthetic sentence variations per rel,
         teaching PERSON (source/target) and REL (role word)

    Also validates cast relationships — drops any whose source or target
    is not in the character whitelist and reports them.

    Returns (examples, known_names, skipped).
    known_names is the set of all UPPER-CASED names registered here —
    used by Layer 2 to block overlapping entities.
    """
    examples    = []
    skipped     = 0
    known_names = set()   # every name registered in this layer

    # ── a) character sentences ─────────────────────────────────────────
    for char in characters:
        name = char["name"] if isinstance(char, dict) else char
        desc = char.get("desc", "") if isinstance(char, dict) else ""
        sentence = f"{name}, {desc}." if desc else f"{name} is a character."
        entities = [(0, len(name), "PERSON")]
        try:
            doc     = nlp.make_doc(sentence)
            example = Example.from_dict(doc, {"entities": entities})
            examples.append(example)
            known_names.add(name.upper())
        except Exception:
            skipped += 1

    # Also register config character names even if not in XML cast
    person_set = (
        {c.upper() for c in config.get("male_characters",  set())} |
        {c.upper() for c in config.get("female_characters", set())}
    )
    known_names |= person_set

    # ── b) relationship sentences ──────────────────────────────────────
    clean_rels  = []
    dropped_rels = []
    for rel in cast_relationships:
        source = rel.get("source", "")
        target = rel.get("target", "")
        source_ok = source.upper() in char_whitelist
        target_ok = target.upper() in char_whitelist
        if source_ok and target_ok:
            clean_rels.append(rel)
        else:
            reason = "source unknown" if not source_ok else "target unknown"
            dropped_rels.append((source, target, reason))

    print(f"\n  [Layer 1 — cast relationships: {play_title}]")
    print(f"    Total : {len(cast_relationships)}  |  "
          f"Clean: {len(clean_rels)}  |  Dropped: {len(dropped_rels)}")
    for src, tgt, reason in dropped_rels:
        print(f"      '{src}' → '{tgt}'  ({reason})")

    for rel in clean_rels:
        source = rel["source"]
        target = rel["target"]
        desc   = rel["description"]

        rel_word_match = re.match(
            r'^(?P<role>.+?)\s+(?:to|of|on|with|unto)\s+', desc, re.IGNORECASE
        )
        rel_word = rel_word_match.group("role").strip() if rel_word_match else None
        if not rel_word:
            continue

        variations = [
            f"{source} is {desc}.",
            f"{source} is the {desc}.",
            f"{target} has a {rel_word} named {source}.",
            f"{source}, {rel_word} of {target}.",
        ]

        for sentence in variations:
            entities = []
            s_start = sentence.find(source)
            if s_start != -1:
                entities.append((s_start, s_start + len(source), "PERSON"))
            r_start = sentence.lower().find(rel_word.lower())
            if r_start != -1:
                entities.append((r_start, r_start + len(rel_word), "REL"))
            t_start = sentence.lower().find(target.lower())
            if t_start != -1:
                entities.append((t_start, t_start + len(target), "PERSON"))

            entities = remove_overlaps(entities)
            if not entities:
                continue
            try:
                doc     = nlp.make_doc(sentence)
                example = Example.from_dict(doc, {"entities": entities})
                examples.append(example)
                known_names.add(source.upper())
                known_names.add(target.upper())
            except Exception:
                skipped += 1

    print(f"    Layer 1 examples built: {len(examples)}  |  "
          f"Known names registered: {len(known_names)}")
    return examples, known_names, skipped


# ── LAYER 2: default NER CSV, filtered by overlap check ──────────────────────

def build_layer2_examples(default_records: list, known_names: set,
                           config: dict, char_whitelist: set,
                           scene_lookup: dict, nlp,
                           play_title: str) -> tuple:
    """
    Layer 2 — supplement Layer 1 with non-character entities from the
    default NER CSV (GPE, LOC, DATE, ORG, etc.).

    For each record:
      1. Correct its label via config (_resolve_label).
      2. OVERLAP CHECK — if entity text contains or is contained by any
         name in known_names, skip it entirely.
         e.g. "MACBETH AND BANQUO" contains "MACBETH" → skipped.
              "WORTHY MACBETH"     contains "MACBETH" → skipped.
      3. Otherwise find every sentence in its scene that contains it
         and create a training example.

    Prints a report of how many records were kept vs overlapped vs dropped.
    """
    person_set = (
        {c.upper() for c in config.get("male_characters",  set())} |
        {c.upper() for c in config.get("female_characters", set())}
    )
    gpe_set   = {g.upper() for g in config.get("known_gpes",      set())}
    loc_set   = {l.upper() for l in config.get("known_locations",  set())}
    title_set = {t.upper() for t in config.get("known_titles",     set())}

    examples     = []
    skipped      = 0
    label_counts = defaultdict(int)
    overlap_dropped  = []
    noise_dropped    = []

    for rec in default_records:
        entity_text = rec.get("entity_text", "").strip()
        old_label   = rec.get("spacy_label", rec.get("label", ""))
        key         = (rec.get("act", ""), rec.get("scene", ""))
        scene_text  = scene_lookup.get(key, "")

        if not entity_text or not scene_text:
            continue

        # Step 1: correct label
        new_label = _resolve_label(
            entity_text, old_label,
            person_set, gpe_set, loc_set, title_set, char_whitelist
        )
        if new_label is None:
            noise_dropped.append(entity_text)
            continue

        # Step 2: overlap check — skip if entity overlaps any Layer 1 name
        if _overlaps_with_known(entity_text, known_names):
            overlap_dropped.append((entity_text, old_label))
            continue

        # Step 3: build examples from real scene sentences
        sentences = re.split(r'(?<=[.!?])\s+', scene_text)
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:
                continue
            spans = find_all_spans(sentence, entity_text)
            if not spans:
                continue
            entities = [(s[0], s[1], new_label) for s in spans]
            entities = remove_overlaps(entities)
            try:
                doc     = nlp.make_doc(sentence)
                example = Example.from_dict(doc, {"entities": entities})
                examples.append(example)
                label_counts[new_label] += 1
            except Exception:
                skipped += 1

    print(f"\n  [Layer 2 — default NER filter: {play_title}]")
    print(f"    Records in      : {len(default_records)}")
    print(f"    Noise dropped   : {len(noise_dropped)}")
    print(f"    Overlap dropped : {len(overlap_dropped)}")
    for entity, label in overlap_dropped:
        print(f"      '{entity}'  [{label}]  overlaps a known name")
    print(f"    Examples built  : {len(examples)}")
    for lbl, cnt in sorted(label_counts.items()):
        print(f"      {lbl:12s}: {cnt} sentences")

    return examples, skipped, label_counts


# ── balance ───────────────────────────────────────────────────────────────────

def balance_examples(all_examples: list) -> list:
    """
    Balance training data so no label dominates.
    Uses median count as target, with a floor of 200.
    """
    label_to_examples = defaultdict(list)
    for ex in all_examples:
        for ent in ex.reference.ents:
            label_to_examples[ent.label_].append(ex)

    print("\n  Label distribution before balancing:")
    for label, exs in sorted(label_to_examples.items()):
        print(f"    [{label:12s}] {len(exs):,} examples")

    counts = sorted(len(v) for v in label_to_examples.values())
    median = counts[len(counts) // 2]
    target = median
    print(f"\n  Target examples per label: {target}")

    balanced  = []
    seen_ids  = set()

    for label, exs in label_to_examples.items():
        if len(exs) >= target:
            sampled = random.sample(exs, target)
        else:
            repeats = (target // len(exs)) + 1
            sampled = (exs * repeats)[:target]

        for ex in sampled:
            if id(ex) not in seen_ids:
                balanced.append(ex)
                seen_ids.add(id(ex))

    print(f"\n  Total after balancing: {len(balanced):,} examples")
    return balanced


# ── STEP 4: fine-tune ──────────────────────────────────────────────────────────

def fine_tune(nlp, all_examples: list, n_iter: int = 40,
              warmup_iters: int = 5,
              lr_start:  float = 0.0002,
              lr_peak:   float = 0.005,
              lr_decay:  float = 0.75,
              decay_every: int = 30):
    """
    Fine-tune NER with a warmup ramp followed by gentle step decay.

    LR schedule
    -----------
    Warmup  (iters 0 → warmup_iters-1):
        LR rises linearly from lr_start → lr_peak.
        Lets the model orient to the new labels before taking big steps.

    Decay   (every decay_every iters after warmup):
        LR multiplied by lr_decay (e.g. 0.75).
        Gentler than the old ×0.5 every 20 — allows more exploration
        before committing to a minimum.

    Default schedule for 160 iters (as seen in your logs):
        0–4   : 0.0002 → 0.005  (warmup)
        30    : 0.005  × 0.75 = 0.00375
        60    : 0.00375× 0.75 = 0.00281
        90    : ...            = 0.00211
        120   : ...            = 0.00158
        150   : ...            = 0.00118
    """
    ner = nlp.get_pipe("ner")

    for label in ["PERSON", "GPE", "LOCATION", "TITLE", "REL"]:
        ner.add_label(label)

    print(f"\nRegistered NER labels: {sorted(ner.labels)}")
    print(f"LR schedule: warmup {warmup_iters} iters "
          f"({lr_start} → {lr_peak}), "
          f"then ×{lr_decay} every {decay_every} iters")

    other_pipes = [p for p in nlp.pipe_names if p != "ner"]
    print(f"Fine-tuning on {len(all_examples):,} examples for {n_iter} iterations ...")

    with nlp.disable_pipes(*other_pipes):
        optimizer            = nlp.resume_training()
        optimizer.learn_rate = lr_start   # start low for warmup

        for i in range(n_iter):

            # ── LR schedule ──────────────────────────────────────────────
            if i < warmup_iters:
                # Linear warmup: lr_start → lr_peak
                progress = (i + 1) / warmup_iters
                optimizer.learn_rate = lr_start + (lr_peak - lr_start) * progress

            elif (i - warmup_iters + 1) % decay_every == 0:
                # Step decay after warmup
                optimizer.learn_rate = max(
                    optimizer.learn_rate * lr_decay,
                    1e-5   # floor — never let LR collapse to zero
                )

            # ── training step ─────────────────────────────────────────────
            random.shuffle(all_examples)
            losses  = {}
            batches = minibatch(all_examples, size=compounding(4.0, 32.0, 1.001))
            for batch in batches:
                nlp.update(batch, drop=0.2, losses=losses)

            # ── logging ───────────────────────────────────────────────────
            if (i + 1) % 10 == 0:
                phase = "warmup" if i < warmup_iters else "train"
                print(f"  Iter {i+1:3d} [{phase:6s}]  "
                      f"NER loss: {losses.get('ner', 0):8.4f}  "
                      f"lr: {optimizer.learn_rate:.6f}")

    return nlp


# ── STEP 5: extract & save ─────────────────────────────────────────────────────

ENTITY_FIELDS = [
    "play", "act", "scene", "location",
    "entity_text", "label", "label_description"
]

def extract_and_save(nlp_ft, scenes: list, play_title: str,
                     stem: str, characters: list, config: dict) -> list:
    """
    Run fine-tuned model over all scenes.

    Two-pass filtering:
      1. PERSON entities must be in the cast whitelist (same as before)
      2. GPE / LOCATION / TITLE entities are cross-checked against config
         known sets — if the config has entries for that play, unlisted
         entities of those types are kept (the model may have found new ones)
         but config-listed ones are guaranteed to appear if detected at all.
    """
    char_whitelist = {
        (c["name"] if isinstance(c, dict) else c).upper()
        for c in characters
    }

    records        = []
    entity_summary = defaultdict(set)
    seen           = set()

    for act_id, scene_id, location, text in scenes:
        doc = nlp_ft(text)
        for ent in doc.ents:
            raw_text   = ent.text.strip()
            normalised = raw_text.upper()
            dedup_key  = (normalised, ent.label_)

            if dedup_key in seen:
                continue

            if ent.label_ == "PERSON" and normalised not in char_whitelist:
                continue

            seen.add(dedup_key)
            records.append({
                "play":              play_title,
                "act":               act_id,
                "scene":             scene_id,
                "location":          location,
                "entity_text":       normalised,
                "label":             ent.label_,
                "label_description": spacy.explain(ent.label_) or ent.label_,
            })
            entity_summary[ent.label_].add(normalised)

    path = Globals.OUTPUT_DIR / f"02_{stem}_finetuned_ner.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ENTITY_FIELDS)
        writer.writeheader()
        writer.writerows(records)

    print(f"  [{play_title}]  {len(records):,} unique entities  → {path.name}")
    for label, ents in sorted(entity_summary.items()):
        print(f"    {label:12s}: {len(ents)} unique")

    return records


# ── MAIN run ───────────────────────────────────────────────────────────────────

def run(nlp_base, play_files: list, cast_relationships: list, n_iter: int = 40):
    all_examples = []
    play_data    = []

    for play_file in play_files:
        print(f"\n{'─'*60}")
        print(f"  Building training data: {play_file.name}")

        root       = DataExtractor.load_play(play_file)[0]
        title      = DataExtractor.get_title(root)
        characters = DataExtractor.extract_cast(root)
        scenes     = DataExtractor.extract_scenes(root)
        play_data.append((play_file, title, characters, scenes))

        config = PLAY_CONFIGS.get(play_file.name, {})
        if not config:
            print(f"  WARNING: No config found for {play_file.name} — "
                  f"falling back to raw spaCy labels")

        # Character whitelist: config sets + XML cast list (merged)
        char_whitelist = (
            {c.upper() for c in config.get("male_characters",  set())} |
            {c.upper() for c in config.get("female_characters", set())} |
            {(c["name"] if isinstance(c, dict) else c).upper() for c in characters}
        )

        # Scene text lookup: (act_id, scene_id) → full text
        scene_lookup = {
            (act_id, scene_id): text
            for act_id, scene_id, _, text in scenes
        }

        # ── LAYER 1: characters + relationships ───────────────────────────
        # Built first — registers all known character names into known_names.
        play_cast_rels = [r for r in cast_relationships if r["play"] == title]
        layer1_examples, known_names, l1_skipped = build_layer1_examples(
            characters, play_cast_rels, char_whitelist, config,
            nlp_base, title
        )
        print(f"  Layer 1 examples   : {len(layer1_examples):,}  "
              f"({l1_skipped} skipped)")

        # ── LAYER 2: default NER CSV, overlap-filtered ────────────────────
        # Loaded second — any entity overlapping a Layer 1 name is dropped.
        default_ner_path = Globals.OUTPUT_DIR / f"01_{play_file.stem}_default_ner.csv"
        if not default_ner_path.exists():
            print(f"  WARNING: Default NER CSV not found at {default_ner_path}")
            print(f"  Run Step 1 (NER Extraction) first.")
            layer2_examples = []
        else:
            import csv as _csv
            with open(default_ner_path, newline="", encoding="utf-8") as f:
                default_records = list(_csv.DictReader(f))
            print(f"  Default NER records loaded: {len(default_records):,}")

            layer2_examples, l2_skipped, label_counts = build_layer2_examples(
                default_records, known_names, config, char_whitelist,
                scene_lookup, nlp_base, title
            )
            print(f"  Layer 2 examples   : {len(layer2_examples):,}  "
                  f"({l2_skipped} skipped)")

        all_examples.extend(layer1_examples)
        all_examples.extend(layer2_examples)

    print(f"\nTotal training examples (before balancing): {len(all_examples):,}")

    balanced_examples = balance_examples(all_examples)
    nlp_ft            = fine_tune(nlp_base, balanced_examples, n_iter=n_iter)

    # Save model
    Globals.MODEL_OUT.mkdir(parents=True, exist_ok=True)
    nlp_ft.to_disk(Globals.MODEL_OUT)
    print(f"\nFine-tuned model saved -> {Globals.MODEL_OUT}")

    # Extract entities with the fine-tuned model
    print("\n=== Extracting entities with fine-tuned model ===")
    all_ft_records = []
    for play_file, title, characters, scenes in play_data:
        config  = PLAY_CONFIGS.get(play_file.name, {})
        records = extract_and_save(
            nlp_ft, scenes, title, play_file.stem, characters, config
        )
        all_ft_records.extend(records)

    # Save combined CSV
    combined_path = Globals.OUTPUT_DIR / "02_ALL_finetuned_ner.csv"
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    with open(combined_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ENTITY_FIELDS)
        writer.writeheader()
        writer.writerows(all_ft_records)
    print(f"  Combined CSV -> {combined_path.name}")

    return nlp_ft, all_ft_records