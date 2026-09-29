"""Generate comprehensive SIGN_CATALOG.txt featuring all Indian languages."""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

with open(PROJECT_ROOT / "translations.json", "r", encoding="utf-8") as f:
    data = json.load(f)

concepts = data.get("concepts", {})

# Descriptions for the 20 concepts
CONCEPT_DETAILS = {
    "hello": {
        "asl": {
            "handshape": "Open flat palm (B-handshape), fingers extended and together.",
            "location": "Temple / forehead (Head zone).",
            "motion": "Start near temple and salute or wave slightly outward forward.",
            "execution": "Dominant hand (one-handed)."
        },
        "isl": {
            "handshape": "Open flat palm facing forward.",
            "location": "Upper chest / shoulder (Head / Chest zone).",
            "motion": "Wave hand side-to-side horizontally.",
            "execution": "Dominant hand (one-handed)."
        }
    },
    "thank_you": {
        "asl": {
            "handshape": "Flat open palm, fingers together pointing upward.",
            "location": "Lips / chin (Chin zone).",
            "motion": "Fingertips lightly touch chin/lips, then extend outward toward conversational partner.",
            "execution": "Dominant hand (one-handed)."
        },
        "isl": {
            "handshape": "Flat open hand touching lips.",
            "location": "Chin / Mouth (Chin zone).",
            "motion": "Touch chin/lips and bring forward slightly, acknowledging receiver.",
            "execution": "Dominant hand (one-handed)."
        }
    },
    "please": {
        "asl": {
            "handshape": "Flat open palm, fingers extended.",
            "location": "Center of the chest (Chest zone).",
            "motion": "Rub flat palm in a gentle clockwise circle against the chest/sternum.",
            "execution": "Dominant hand (one-handed)."
        },
        "isl": {
            "handshape": "Both open palms pressed together vertically (Namaste / Prayer).",
            "location": "Center of chest (Chest zone).",
            "motion": "Held still or dipped slightly in traditional Indian respect gesture.",
            "execution": "Both hands (two-handed)."
        }
    },
    "sorry": {
        "asl": {
            "handshape": "Closed fist (A-handshape, thumb across fingers).",
            "location": "Over the heart / chest (Chest zone).",
            "motion": "Rub closed fist in a circular motion on the chest.",
            "execution": "Dominant hand (one-handed)."
        },
        "isl": {
            "handshape": "Fingers grasping or holding earlobe (ISL traditional ear-hold).",
            "location": "Ear level (Head zone).",
            "motion": "Grasp earlobes with thumb and index fingers, holding head slightly bowed.",
            "execution": "Single hand or both hands at ears."
        }
    },
    "pain": {
        "asl": {
            "handshape": "Both hands with index fingers extended (1-handshape), pointing tips toward each other.",
            "location": "Chest or location of injury (Chest zone).",
            "motion": "Twist or jab index fingers repeatedly toward each other without touching.",
            "execution": "Both hands (two-handed)."
        },
        "isl": {
            "handshape": "Index finger extended.",
            "location": "Chest or area of distress.",
            "motion": "Repeated jabbing or pointing at chest or pain location.",
            "execution": "Dominant hand or two hands."
        }
    },
    "no": {
        "asl": {
            "handshape": "Index and middle finger snap quickly down onto thumb.",
            "location": "Chin / neutral zone.",
            "motion": "Rapid closing snap of index and middle fingers against thumb.",
            "execution": "Dominant hand (one-handed)."
        },
        "isl": {
            "handshape": "Index finger pointing upward.",
            "location": "Chin / chest zone.",
            "motion": "Wag index finger side-to-side in clear negation.",
            "execution": "Dominant hand (one-handed)."
        }
    },
    "yes": {
        "asl": {
            "handshape": "Closed fist (S-handshape).",
            "location": "In front of chest / neutral zone.",
            "motion": "Fist nods up and down pivoting at wrist like a head nodding.",
            "execution": "Dominant hand (one-handed)."
        },
        "isl": {
            "handshape": "Closed fist.",
            "location": "In front of chest.",
            "motion": "Vertical nodding or dipping of the fist.",
            "execution": "Dominant hand (one-handed)."
        }
    },
    "help": {
        "asl": {
            "handshape": "Non-dominant flat open palm facing up; dominant hand thumbs-up resting on palm.",
            "location": "Chest level.",
            "motion": "Both hands lift upward together.",
            "execution": "Both hands (two-handed)."
        },
        "isl": {
            "handshape": "Dominant thumbs-up placed on non-dominant open flat palm.",
            "location": "Chest level.",
            "motion": "Upward lifting motion indicating support.",
            "execution": "Both hands (two-handed)."
        }
    },
    "water": {
        "asl": {
            "handshape": "W-handshape (Index, Middle, Ring extended; thumb over pinky).",
            "location": "Chin / Lips (Chin zone).",
            "motion": "Tap index finger of W-hand twice against the chin.",
            "execution": "Dominant hand (one-handed)."
        },
        "isl": {
            "handshape": "Cup hand or W-handshape near mouth.",
            "location": "Mouth / Chin zone.",
            "motion": "Tilting toward mouth simulating drinking.",
            "execution": "Dominant hand (one-handed)."
        }
    },
    "food": {
        "asl": {
            "handshape": "Bunched fingers (Flat-O: all fingertips touching thumb).",
            "location": "Mouth / Lips (Chin zone).",
            "motion": "Tap bunched fingertips twice against the lips.",
            "execution": "Dominant hand (one-handed)."
        },
        "isl": {
            "handshape": "Bunched fingertips touching thumb.",
            "location": "Mouth (Chin zone).",
            "motion": "Bringing food to mouth motion repeatedly.",
            "execution": "Dominant hand (one-handed)."
        }
    },
    "medicine": {
        "asl": {
            "handshape": "Non-dominant hand flat open palm up; dominant middle finger extended.",
            "location": "Chest level.",
            "motion": "Middle fingertip twists or presses into the center of the open palm.",
            "execution": "Both hands (two-handed)."
        },
        "isl": {
            "handshape": "Fingertip grinding or pressing in palm.",
            "location": "Chest level.",
            "motion": "Stirring or grinding motion representing crushing medicine.",
            "execution": "Both hands (two-handed)."
        }
    },
    "doctor": {
        "asl": {
            "handshape": "Dominant index and middle fingertips (M-handshape).",
            "location": "Inner wrist of non-dominant arm (Chest zone).",
            "motion": "Tap fingertips twice on inner wrist taking radial pulse.",
            "execution": "Both hands (two-handed)."
        },
        "isl": {
            "handshape": "Fingertips touching opposite wrist.",
            "location": "Radial pulse / wrist (Chest zone).",
            "motion": "Touching inner wrist checking pulse.",
            "execution": "Both hands (two-handed)."
        }
    },
    "family": {
        "asl": {
            "handshape": "Both hands in F-handshapes (thumb and index touching, 3 fingers up).",
            "location": "Chest level.",
            "motion": "Both hands sweep outward in a circle and meet with pinkies touching.",
            "execution": "Both hands (two-handed)."
        },
        "isl": {
            "handshape": "Both open hands forming a circle.",
            "location": "Chest level.",
            "motion": "Enclosing circular embrace motion.",
            "execution": "Both hands (two-handed)."
        }
    },
    "work": {
        "asl": {
            "handshape": "Both hands in closed fists.",
            "location": "Chest level.",
            "motion": "Dominant fist taps the wrist/back of non-dominant fist twice.",
            "execution": "Both hands (two-handed)."
        },
        "isl": {
            "handshape": "Both fists tapping together.",
            "location": "Chest level.",
            "motion": "Rhythmic tapping of fists together.",
            "execution": "Both hands (two-handed)."
        }
    },
    "school": {
        "asl": {
            "handshape": "Both hands open palms.",
            "location": "Chest level.",
            "motion": "Dominant flat palm claps down horizontally twice onto non-dominant flat palm.",
            "execution": "Both hands (two-handed)."
        },
        "isl": {
            "handshape": "Both open hands.",
            "location": "Chest level.",
            "motion": "Horizontal clapping motion representing books / learning.",
            "execution": "Both hands (two-handed)."
        }
    },
    "home": {
        "asl": {
            "handshape": "Bunched fingers (Flat-O shape).",
            "location": "Cheek near mouth, then cheek near ear.",
            "motion": "Touch cheek near mouth, then touch cheek near ear (eat + sleep).",
            "execution": "Dominant hand (one-handed)."
        },
        "isl": {
            "handshape": "Both hands forming roof / inverted V.",
            "location": "Chest / chin level.",
            "motion": "Fingertips touching at angle like house roof.",
            "execution": "Both hands or cheek touch."
        }
    },
    "money": {
        "asl": {
            "handshape": "Dominant hand bunched fingers.",
            "location": "Non-dominant flat palm or chest.",
            "motion": "Tap back of bunched fingers into palm or rub thumb across fingertips.",
            "execution": "Dominant hand or two hands."
        },
        "isl": {
            "handshape": "Thumb rubbing across index and middle fingertips.",
            "location": "Chest / neutral level.",
            "motion": "Fingers rubbing together indicating counting coins/rupees.",
            "execution": "Dominant hand (one-handed)."
        }
    },
    "phone": {
        "asl": {
            "handshape": "Y-handshape (Thumb and Pinky extended, middle three curled).",
            "location": "Side of head / ear (Head zone).",
            "motion": "Thumb near ear, pinky near mouth, held or tilted.",
            "execution": "Dominant hand (one-handed)."
        },
        "isl": {
            "handshape": "Y-handshape or fist to ear.",
            "location": "Ear level (Head zone).",
            "motion": "Holding phone receiver to ear.",
            "execution": "Dominant hand (one-handed)."
        }
    },
    "happy": {
        "asl": {
            "handshape": "Flat open hand, fingers together, palm facing chest.",
            "location": "Chest / sternum level.",
            "motion": "Brush palm upward against chest in repeated cheerful upward strokes.",
            "execution": "One or both hands."
        },
        "isl": {
            "handshape": "Open flat hands brushing upward.",
            "location": "Chest level.",
            "motion": "Upward brushing stroke conveying uplifting joy.",
            "execution": "One or both hands."
        }
    },
    "sad": {
        "asl": {
            "handshape": "Open hand with slightly drooping fingers facing inward.",
            "location": "Eyes down to chin (Chin zone).",
            "motion": "Hand drags downward across the face while head tilts down.",
            "execution": "Dominant hand (one-handed)."
        },
        "isl": {
            "handshape": "Open hand drooping downward.",
            "location": "Face down to chest.",
            "motion": "Downward drawing motion symbolizing sorrow.",
            "execution": "Dominant hand (one-handed)."
        }
    },
}

LANG_DISPLAY = [
    ("hi", "Hindi (हिंदी)"),
    ("ta", "Tamil (தமிழ்)"),
    ("te", "Telugu (తెలుగు)"),
    ("bn", "Bengali (বাংলা)"),
    ("mr", "Marathi (मराठी)"),
    ("gu", "Gujarati (ગુજરાતી)"),
    ("kn", "Kannada (ಕನ್ನಡ)"),
    ("ml", "Malayalam (മലയാളം)"),
    ("pa", "Punjabi (ਪੰਜਾਬੀ)"),
    ("ur", "Urdu (اردو)"),
    ("en", "English (Pivot)"),
]

lines = []
lines.append("="*80)
lines.append("          OFFLINE DUAL-LANGUAGE SIGN LANGUAGE RECOGNITION CATALOG")
lines.append("                  ISL (Indian) & ASL (American) Signs")
lines.append("="*80)
lines.append("")
lines.append("Total Classes: 40 (20 ISL Classes + 20 ASL Classes)")
lines.append("Supported Target Languages (All Major Indian Languages):")
for code, name in LANG_DISPLAY:
    lines.append(f"  - {name} [{code}]")
lines.append("")

# ASL Classes (20-39)
lines.append("-" * 80)
lines.append("PART 1: AMERICAN SIGN LANGUAGE (ASL) SIGNS (Classes 20 to 39)")
lines.append("-" * 80)

for idx, (concept, details) in enumerate(CONCEPT_DETAILS.items(), start=20):
    asl_det = details["asl"]
    lines.append("")
    lines.append(f"[Class {idx}] {concept.upper().replace('_', ' ')}")
    lines.append(f"- Concept: {concept}")
    lines.append(f"- Handshape: {asl_det['handshape']}")
    lines.append(f"- Location: {asl_det['location']}")
    lines.append(f"- Motion: {asl_det['motion']}")
    lines.append(f"- Execution: {asl_det['execution']}")
    lines.append("- Translations across Indian Languages:")
    c_dict = concepts.get(concept, {})
    for code, name in LANG_DISPLAY:
        entry = c_dict.get(code, "")
        txt = entry.get("text", "") if isinstance(entry, dict) else str(entry)
        lines.append(f"  * {name}: {txt}")

# ISL Classes (0-19)
lines.append("")
lines.append("-" * 80)
lines.append("PART 2: INDIAN SIGN LANGUAGE (ISL) SIGNS (Classes 0 to 19)")
lines.append("-" * 80)

for idx, (concept, details) in enumerate(CONCEPT_DETAILS.items(), start=0):
    isl_det = details["isl"]
    lines.append("")
    lines.append(f"[Class {idx}] {concept.upper().replace('_', ' ')}")
    lines.append(f"- Concept: {concept}")
    lines.append(f"- Handshape: {isl_det['handshape']}")
    lines.append(f"- Location: {isl_det['location']}")
    lines.append(f"- Motion: {isl_det['motion']}")
    lines.append(f"- Execution: {isl_det['execution']}")
    lines.append("- Translations across Indian Languages:")
    c_dict = concepts.get(concept, {})
    for code, name in LANG_DISPLAY:
        entry = c_dict.get(code, "")
        txt = entry.get("text", "") if isinstance(entry, dict) else str(entry)
        lines.append(f"  * {name}: {txt}")

lines.append("")
lines.append("=" * 80)
lines.append("END OF CATALOG - ALL 40 SIGNS & 11 INDIAN LANGUAGES")
lines.append("=" * 80)

catalog_txt = "\n".join(lines)
catalog_path = PROJECT_ROOT / "SIGN_CATALOG.txt"
with open(catalog_path, "w", encoding="utf-8") as f:
    f.write(catalog_txt)

print(f"Generated complete SIGN_CATALOG.txt ({len(catalog_txt)} characters).")
