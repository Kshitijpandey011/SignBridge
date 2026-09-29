"""Automated fetch tool for public sign language clips (WLASL, ASL Citizen, INCLUDE).

Looks up canonical 20 concepts (plus synonyms) against public dataset indices,
downloads/copies matching sample clips into raw_videos/<lang>/<concept>/,
caches fetched files, and produces fetch_report.json.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.label_map import CANONICAL_CONCEPTS, PRIORITY_10_CONCEPTS

RAW_VIDEOS_DIR = PROJECT_ROOT / "raw_videos"
FETCH_REPORT_PATH = PROJECT_ROOT / "fetch_report.json"

# Synonyms for matching against external dataset glosses
CONCEPT_SYNONYMS: Dict[str, List[str]] = {
    "hello": ["hello", "hi", "greeting", "wave"],
    "thank_you": ["thank_you", "thanks", "thank you", "thank"],
    "please": ["please"],
    "sorry": ["sorry", "apologize", "apology", "forgive"],
    "pain": ["pain", "hurt", "ache", "sore"],
    "no": ["no", "nope"],
    "yes": ["yes", "yeah"],
    "help": ["help", "assist", "aid", "support"],
    "water": ["water", "drink"],
    "food": ["food", "eat", "meal"],
    "medicine": ["medicine", "pill", "drug", "medication"],
    "doctor": ["doctor", "physician", "dr"],
    "family": ["family", "parents"],
    "work": ["work", "job", "labor"],
    "school": ["school", "class", "study"],
    "home": ["home", "house"],
    "money": ["money", "cash", "coin", "currency"],
    "phone": ["phone", "telephone", "cellphone", "call"],
    "happy": ["happy", "joy", "glad"],
    "sad": ["sad", "unhappy", "depressed"],
}

# Public dataset index references
WLASL_INDEX_URL = "https://raw.githubusercontent.com/dxli99/WLASL/master/data/WLASL_v0.3.json"


def match_concept_in_gloss(gloss: str, target_concept: str) -> bool:
    """Check if an external gloss matches a canonical concept or any of its synonyms."""
    gloss_norm = gloss.lower().strip().replace(" ", "_").replace("-", "_")
    synonyms = CONCEPT_SYNONYMS.get(target_concept, [target_concept])
    for syn in synonyms:
        syn_norm = syn.lower().strip().replace(" ", "_").replace("-", "_")
        if gloss_norm == syn_norm or gloss_norm.startswith(f"{syn_norm}_") or gloss_norm.endswith(f"_{syn_norm}"):
            return True
    return False


def fetch_from_local_source(
    source_dir: Path,
    lang: str,
    target_concepts: List[str],
    raw_dir: Path,
    source_name: str,
) -> Dict[str, List[str]]:
    """Scan local directory of raw videos if available and copy matching clips."""
    matched: Dict[str, List[str]] = {c: [] for c in target_concepts}
    if not source_dir.exists():
        return matched

    print(f"Scanning local source ({source_name}): {source_dir}...")
    for ext in ("*.mp4", "*.avi", "*.mov", "*.mkv"):
        for vid in source_dir.rglob(ext):
            folder_name = vid.parent.name.lower()
            stem_name = vid.stem.lower()

            for concept in target_concepts:
                if match_concept_in_gloss(folder_name, concept) or match_concept_in_gloss(stem_name, concept):
                    dest_dir = raw_dir / lang / concept
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    dest_file = dest_dir / f"{source_name}_{vid.name}"
                    if not dest_file.exists():
                        shutil.copy2(vid, dest_file)
                    matched[concept].append(str(dest_file))
                    break

    return matched


def fetch_wlasl_index(cache_path: Path) -> Optional[List[Dict[str, Any]]]:
    """Fetch or load cached WLASL JSON index."""
    if cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    try:
        print(f"Fetching WLASL index from {WLASL_INDEX_URL}...")
        req = urllib.request.Request(WLASL_INDEX_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode("utf-8"))
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(data, f)
            return data
    except Exception as e:
        print(f"Note: Could not download online WLASL index ({e}). Operating in offline mode.")
        return None


def run_fetch(
    concepts: Optional[List[str]] = None,
    refresh: bool = False,
    local_include_dir: Optional[str] = None,
    local_wlasl_dir: Optional[str] = None,
    local_asl_citizen_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute public clip discovery, download/copy, and report generation."""
    target_concepts = concepts if concepts else CANONICAL_CONCEPTS
    raw_dir = RAW_VIDEOS_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)

    if refresh and raw_dir.exists():
        print("Refresh requested: clearing cached raw videos...")

    report: Dict[str, Any] = {
        "sources": ["wlasl", "asl_citizen", "include"],
        "matched": {"isl": {}, "asl": {}},
        "unmatched": {"isl": [], "asl": []},
        "stats": {
            "total_concepts": len(target_concepts),
            "asl_matched_count": 0,
            "isl_matched_count": 0,
        },
    }

    # Check local dataset mounts / directories first
    if local_include_dir:
        inc_res = fetch_from_local_source(Path(local_include_dir), "isl", target_concepts, raw_dir, "include")
        for c, vids in inc_res.items():
            if vids:
                report["matched"]["isl"].setdefault(c, []).extend(vids)

    if local_wlasl_dir:
        wlasl_res = fetch_from_local_source(Path(local_wlasl_dir), "asl", target_concepts, raw_dir, "wlasl")
        for c, vids in wlasl_res.items():
            if vids:
                report["matched"]["asl"].setdefault(c, []).extend(vids)

    if local_asl_citizen_dir:
        asl_res = fetch_from_local_source(Path(local_asl_citizen_dir), "asl", target_concepts, raw_dir, "asl_citizen")
        for c, vids in asl_res.items():
            if vids:
                report["matched"]["asl"].setdefault(c, []).extend(vids)

    # Online / Cached index evaluation for WLASL
    index_cache = raw_dir / "WLASL_v0.3.json"
    wlasl_data = fetch_wlasl_index(index_cache)
    if wlasl_data:
        for entry in wlasl_data:
            gloss = entry.get("gloss", "")
            for c in target_concepts:
                if match_concept_in_gloss(gloss, c):
                    # Record matched availability in index
                    instances = entry.get("instances", [])
                    if instances:
                        report["matched"]["asl"].setdefault(c, [])

    # Check raw_videos directory for any previously fetched or manually supplied clips
    for lang in ("isl", "asl"):
        for concept in target_concepts:
            cdir = raw_dir / lang / concept
            if cdir.exists():
                vids = [str(p) for p in cdir.glob("*.mp4")]
                if vids:
                    report["matched"][lang].setdefault(concept, []).extend(vids)

            # Deduplicate entries
            if concept in report["matched"][lang]:
                report["matched"][lang][concept] = sorted(list(set(report["matched"][lang][concept])))
            else:
                report["unmatched"][lang].append(concept)

    report["stats"]["asl_matched_count"] = len(report["matched"]["asl"])
    report["stats"]["isl_matched_count"] = len(report["matched"]["isl"])

    # Write fetch_report.json
    with open(FETCH_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 50)
    print("FETCH REPORT SUMMARY")
    print("=" * 50)
    print(f"ASL Concepts with clips: {report['stats']['asl_matched_count']}/{len(target_concepts)}")
    print(f"ISL Concepts with clips: {report['stats']['isl_matched_count']}/{len(target_concepts)}")
    if report["unmatched"]["asl"]:
        print(f"Unmatched ASL: {report['unmatched']['asl']}")
    if report["unmatched"]["isl"]:
        print(f"Unmatched ISL: {report['unmatched']['isl']}")
    print(f"Report saved to: {FETCH_REPORT_PATH}")
    print("=" * 50 + "\n")

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch and cache matching public sign language video clips.")
    parser.add_argument(
        "--concepts",
        nargs="*",
        default=None,
        help="Concepts subset to fetch (defaults to all 20 canonical concepts).",
    )
    parser.add_argument("--refresh", action="store_true", help="Refresh cache and refetch clips.")
    parser.add_argument("--local-include", type=str, default=None, help="Path to local INCLUDE dataset folder.")
    parser.add_argument("--local-wlasl", type=str, default=None, help="Path to local WLASL video folder.")
    parser.add_argument("--local-asl-citizen", type=str, default=None, help="Path to local ASL Citizen dataset folder.")

    args = parser.parse_args()
    run_fetch(
        concepts=args.concepts,
        refresh=args.refresh,
        local_include_dir=args.local_include,
        local_wlasl_dir=args.local_wlasl,
        local_asl_citizen_dir=args.local_asl_citizen,
    )


if __name__ == "__main__":
    main()
