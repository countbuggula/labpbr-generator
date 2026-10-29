#!/usr/bin/env python3
"""
LabPBR 1.3 Automated Add-on Generator for Minecraft Resource Packs (Any Resolution).
Preserves original sprites 100% by outputting a standalone overlay resource pack
containing only pack.mcmeta, *_n.png, *_s.png, and animation *.mcmeta files.
Supports blocks, items/weapons/tools, 3D armor/equipment, mobs/creatures/animals/villagers,
paintings (auto-detected polished wood frame + flat satin oil canvas), and banners.
"""

import argparse
import base64
import io
import json
import os
import re
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests
from PIL import Image


# =====================================================================
# LabPBR 1.3 Reference Constants
# =====================================================================
# Green channel (G) in _s.png:
#   0-229   : Dielectric F0 reflectance (e.g., 6 = wood, 10 = stone, 26 = gem)
#   230-237 : Hardcoded LabPBR metals (Iron, Gold, Aluminum, Chrome, Copper, Lead, Platinum, Silver)
#   255     : Albedo-tinted metal (uses the base texture color as metallic tint)
LABPBR_METALS = {
    "iron": 230,
    "gold": 231,
    "aluminum": 232,
    "chrome": 233,
    "copper": 234,
    "lead": 235,
    "platinum": 236,
    "silver": 237,
    "tinted_metal": 255,
}

VALID_CATEGORIES = {
    "stone",
    "wood",
    "metal",
    "gem",
    "ore",
    "foliage",
    "emissive",
    "glass",
    "soft",
    "composite",
    "painting",
    "banner",
    "tool",
    "mob",
}
VALID_MASK_MODES = {
    "none",
    "color_outlier",
    "high_saturation",
    "bright_pixels",
    "dark_pixels",
    "wood_handle",
}

# Valid vanilla Minecraft AnimationMetadataSection fields (.png.mcmeta)
VALID_ANIMATION_KEYS = {"frametime", "interpolate", "frames", "width", "height"}

# Default resource pack texture directories scanned for PBR generation
# Note: 'assets/minecraft/textures/entity' recursively covers all mobs, animals,
# villagers, banners, chests, beds, boats, shields, and 3D equipment!
DEFAULT_TEXTURE_SUBDIRS: Tuple[str, ...] = (
    "assets/minecraft/textures/block",
    "assets/minecraft/textures/blocks",
    "assets/minecraft/textures/item",
    "assets/minecraft/textures/items",
    "assets/minecraft/textures/painting",
    "assets/minecraft/textures/paintings",
    "assets/minecraft/textures/entity",
    "assets/minecraft/textures/models/armor",
    "assets/minecraft/optifine/random/entity",
)

# Soft transition width above mask_threshold and discrete F0 step cutoff
MASK_RAMP_WIDTH = 0.20
ACCENT_STEP_CUTOFF = 0.20

# Persistent HTTP session for connection pooling with local Ollama
HTTP_SESSION = requests.Session()

OLLAMA_OPTIONS = {
    "temperature": 0.1,
    "num_ctx": 2048,
    "num_predict": 256,
}

OLLAMA_SCHEMA = {
    "type": "object",
    "properties": {
        "category": {
            "type": "string",
            "enum": ["stone", "wood", "metal", "gem", "ore", "foliage", "emissive", "glass", "soft", "composite", "tool", "mob"],
        },
        "normal_strength": {"type": "number", "description": "0.5 (flat/smooth) to 4.0 (deep cobble/bricks)"},
        "height_contrast": {"type": "number", "description": "0.05 (flat) to 0.45 (deep relief)"},
        "invert_height": {"type": "boolean", "description": "True if darker pixels are physically raised"},
        "base_smoothness": {"type": "integer", "description": "0 (matte) to 255 (mirror polished)"},
        "base_f0": {"type": "integer", "description": "0-229 dielectric, 230-237 LabPBR metal ID, or 255 tinted metal"},
        "porosity_or_sss": {
            "type": "integer",
            "description": "0-64 for surface porosity (stone/soil/wood/fur), 65-255 for subsurface scattering (leaves/snow/slime/aquatic)",
        },
        "mask_mode": {
            "type": "string",
            "enum": ["none", "color_outlier", "high_saturation", "bright_pixels", "dark_pixels", "wood_handle"],
            "description": "How to isolate secondary/accent pixels like ore gems, glowing cracks, or wooden tool handles",
        },
        "mask_threshold": {"type": "number", "description": "Sensitivity from 0.15 (broad) to 0.75 (strict)"},
        "accent_smoothness": {"type": "integer", "description": "0-255 smoothness for masked accent pixels"},
        "accent_f0": {"type": "integer", "description": "0-255 F0/metal ID for masked accent pixels"},
        "accent_emissive": {"type": "integer", "description": "0 (no glow) to 254 (max glow) for masked accent pixels"},
        "accent_height_offset": {"type": "number", "description": "-0.25 (recessed) to +0.25 (protruding) for accent pixels"},
    },
    "required": [
        "category",
        "normal_strength",
        "height_contrast",
        "invert_height",
        "base_smoothness",
        "base_f0",
        "porosity_or_sss",
        "mask_mode",
        "mask_threshold",
        "accent_smoothness",
        "accent_f0",
        "accent_emissive",
        "accent_height_offset",
    ],
}


# =====================================================================
# Pack Resolution & Painting Frame Border Detection
# =====================================================================
def detect_pack_resolution(input_dir: Path) -> int:
    """
    Detects the base tile resolution (e.g., 16, 32, 64, 128, 256, 512) of the input
    resource pack by sampling square block/item textures and finding the modal width.
    """
    widths: List[int] = []
    for sub in ("assets/minecraft/textures/block", "assets/minecraft/textures/blocks", "assets/minecraft/textures/item"):
        folder = input_dir / sub
        if not folder.is_dir():
            continue
        for root, _, files in os.walk(folder):
            root_path = Path(root)
            for f in files:
                fl = f.lower()
                if not fl.endswith(".png") or fl.endswith(("_n.png", "_s.png", "_e.png")):
                    continue
                try:
                    with Image.open(root_path / f) as im:
                        w, h = im.size
                        if w > 0 and (w == h or (h > w and h % w == 0)):
                            widths.append(w)
                except Exception:
                    continue
                if len(widths) >= 80:
                    break
            if len(widths) >= 80:
                break
        if widths:
            break

    if not widths:
        return 16
    return Counter(widths).most_common(1)[0][0]


def detect_pack_painting_border(painting_dir: Path) -> Optional[int]:
    """
    Automatically detects the wooden frame border thickness (in pixels) shared across
    paintings in a resource pack by measuring cross-painting pixel consistency (MAD)
    along concentric border rings d = 0, 1, 2, ...
    """
    if not painting_dir.is_dir():
        return None

    arrays: List[np.ndarray] = []
    for f in sorted(painting_dir.glob("*.png")):
        fl = f.name.lower()
        if fl == "back.png" or fl.endswith(("_n.png", "_s.png", "_e.png")):
            continue
        try:
            with Image.open(f) as im:
                rgba = np.array(im.convert("RGBA"))
                if np.mean(rgba[..., 3] > 16) < 0.8:
                    continue
                arrays.append(rgba[..., :3].astype(np.float32))
        except Exception:
            continue
        if len(arrays) >= 20:
            break

    if len(arrays) < 2:
        return None

    min_dim = min(min(a.shape[0], a.shape[1]) for a in arrays)
    max_search = max(1, min(min_dim // 4, 32))
    span_len = max(4, min_dim // 2)

    border_width = 0
    for d in range(max_search):
        if d + span_len >= min_dim:
            break
        strips = np.array([
            np.concatenate([a[d, d : d + span_len], a[d : d + span_len, d]], axis=0)
            for a in arrays
        ])
        med = np.median(strips, axis=0)
        mad = float(np.mean(np.abs(strips - med)))
        if mad < 16.0:
            border_width = d + 1
        else:
            break

    return border_width if border_width > 0 else None


def detect_single_painting_border(rgb_float: np.ndarray, pack_border: Optional[int] = None) -> int:
    """
    Determines the frame border thickness in pixels for a single painting image.
    Uses pack_border if valid, or falls back to analyzing tangential edge continuity
    along concentric rectangular rings.
    """
    h, w, _ = rgb_float.shape
    max_d = max(1, min(h, w) // 5)

    if pack_border is not None and 1 <= pack_border <= max_d:
        return pack_border

    tangential_diffs: List[float] = []
    for d in range(max_d + 1):
        if w - 2 * (d + 1) < 2 or h - 2 * (d + 1) < 2:
            break
        sx = slice(d + 1, w - d - 1)
        sy = slice(d + 1, h - d - 1)
        top_dx = float(np.mean(np.linalg.norm(np.diff(rgb_float[d, sx], axis=0), axis=-1)))
        bot_dx = float(np.mean(np.linalg.norm(np.diff(rgb_float[h - 1 - d, sx], axis=0), axis=-1)))
        left_dy = float(np.mean(np.linalg.norm(np.diff(rgb_float[sy, d], axis=0), axis=-1)))
        right_dy = float(np.mean(np.linalg.norm(np.diff(rgb_float[sy, w - 1 - d], axis=0), axis=-1)))
        tangential_diffs.append((top_dx + bot_dx + left_dy + right_dy) / 4.0)

    if len(tangential_diffs) >= 2:
        base_ref = max(tangential_diffs[0], 0.015)
        for d in range(1, len(tangential_diffs)):
            if tangential_diffs[d] > max(base_ref * 5.0, 0.14):
                return d

    unit = min(h, w)
    while unit > 64 and unit % 2 == 0:
        unit //= 2
    return max(1, round(unit / 16))


# =====================================================================
# Deterministic Fallback Heuristics & Vision LLM Profiler
# =====================================================================
def sanitize_f0(value: Any, default: int = 10) -> int:
    """
    Clamps an F0 / metalness value to valid LabPBR 1.3 Green-channel ranges:
      0-229   : Dielectric F0 reflectance
      230-237 : Predefined LabPBR metals
      255     : Albedo-tinted metal
    Values in the unallocated 238-254 gap are snapped to 237 or 255.
    """
    try:
        v = int(round(float(value)))
    except (TypeError, ValueError):
        return default
    v = max(0, min(255, v))
    if 238 <= v <= 254:
        return 237 if v < 246 else 255
    return v


def _clamp_float(value: Any, low: float, high: float, default: float) -> float:
    try:
        return float(max(low, min(high, float(value))))
    except (TypeError, ValueError):
        return default


def _clamp_int(value: Any, low: int, high: int, default: int) -> int:
    try:
        return int(max(low, min(high, int(round(float(value))))))
    except (TypeError, ValueError):
        return default


def sanitize_profile(
    profile: Dict[str, Any],
    fallback: Optional[Dict[str, Any]] = None,
    source: Optional[str] = None,
) -> Dict[str, Any]:
    """Ensures all profile fields exist and fall strictly within valid LabPBR 1.3 ranges."""
    fb = fallback if fallback is not None else heuristic_profile("stone")
    if not isinstance(profile, dict):
        return dict(fb)

    category = str(profile.get("category", fb["category"]))
    if category not in VALID_CATEGORIES:
        category = fb["category"]

    mask_mode = str(profile.get("mask_mode", fb["mask_mode"]))
    if mask_mode not in VALID_MASK_MODES:
        mask_mode = fb["mask_mode"]

    base_smooth = _clamp_int(profile.get("base_smoothness"), 0, 255, fb["base_smoothness"])
    base_f0 = sanitize_f0(profile.get("base_f0"), fb["base_f0"])
    accent_f0 = sanitize_f0(profile.get("accent_f0"), fb["accent_f0"])

    # Guard against LLM placing metal ID into base_f0 on stone-hosted ore blocks
    if category == "ore" and base_f0 >= 230:
        if accent_f0 < 230:
            accent_f0 = base_f0
        base_f0 = fb["base_f0"]
        base_smooth = min(base_smooth, 75)

    # Guard against LLM assigning solid metal F0 to organic mobs/animals/villagers
    if fb.get("category") == "mob" and fb["base_f0"] < 230 and base_f0 >= 230:
        base_f0 = fb["base_f0"]
        base_smooth = min(base_smooth, 190)

    # Guard against LLM stripping wooden handle masking or shiny metal F0 on tools/weapons
    if fb.get("category") == "tool" and fb.get("mask_mode") == "wood_handle":
        category = "tool"
        mask_mode = "wood_handle"
        if fb["base_f0"] >= 230 and base_f0 < 230:
            base_f0 = fb["base_f0"]
            base_smooth = max(base_smooth, 200)
        accent_f0 = 6
        accent_smooth = 75
    else:
        accent_smooth = _clamp_int(profile.get("accent_smoothness"), 0, 255, fb["accent_smoothness"])

    resolved_source = source if source is not None else str(profile.get("source", fb.get("source", "heuristic")))

    return {
        "category": category,
        "normal_strength": _clamp_float(profile.get("normal_strength"), 0.4, 5.0, fb["normal_strength"]),
        "height_contrast": _clamp_float(profile.get("height_contrast"), 0.04, 0.6, fb["height_contrast"]),
        "invert_height": bool(profile.get("invert_height", fb["invert_height"])),
        "base_smoothness": base_smooth,
        "base_f0": base_f0,
        "porosity_or_sss": _clamp_int(profile.get("porosity_or_sss"), 0, 255, fb["porosity_or_sss"]),
        "mask_mode": mask_mode,
        "mask_threshold": _clamp_float(profile.get("mask_threshold"), 0.05, 0.90, fb["mask_threshold"]),
        "accent_smoothness": accent_smooth,
        "accent_f0": accent_f0,
        "accent_emissive": _clamp_int(profile.get("accent_emissive"), 0, 254, fb["accent_emissive"]),
        "accent_height_offset": _clamp_float(profile.get("accent_height_offset"), -0.35, 0.35, fb["accent_height_offset"]),
        "source": resolved_source,
    }


def is_llm_profile(profile: Optional[Dict[str, Any]]) -> bool:
    """Returns True only if the cached profile was actually generated by an Ollama VLM or deterministic special rule."""
    if not isinstance(profile, dict):
        return False
    src = str(profile.get("source", ""))
    return src.startswith("llm:") or src.startswith("builtin:")


def get_material_group_id(rel_subdir: str, rel_block_id: str) -> str:
    """
    Groups numbered random variant textures or animal/villager coat/biome variants into a
    shared material group so the VLM only needs to profile one representative sprite per
    material group while every variant still gets its own pixel-exact _n and _s maps.
    """
    sub_lower = rel_subdir.lower()
    rid = rel_block_id.lower()

    if "/" in rid:
        parent, stem = rid.rsplit("/", 1)
        # Numbered variants (e.g., 'ore/gold/0', 'optifine/random/entity/cow/cow2')
        if stem.isdigit() or re.match(r"^[a-z_]+\d+$", stem):
            return f"{sub_lower}:{parent}"

        # Entity coat, color, and biome variants that share identical surface physics
        if "entity" in sub_lower:
            if parent in (
                "villager/type", "villager/baby", "zombie_villager/type", "zombie_villager/baby",
                "cat", "frog", "axolotl", "rabbit", "fox", "panda", "parrot", "llama", "goat",
                "cow", "pig", "chicken", "sheep", "turtle", "dolphin", "bat", "camel", "sniffer",
                "cushion", "boat", "chest_boat", "shulker",
            ):
                return f"{sub_lower}:{parent}"
            if parent == "wolf" and "armor" not in stem and "collar" not in stem:
                return f"{sub_lower}:wolf_coat"
            if parent == "horse" and "armor" not in stem and "skeleton" not in stem and "zombie" not in stem:
                return f"{sub_lower}:horse_coat"

    return f"{sub_lower}:{rid}"


def verify_ollama_connection(ollama_url: str, model: str) -> bool:
    """
    Pre-flight check that verifies the Ollama server is reachable, checks that the target
    vision model is installed, and performs a warm-up request so cold-start VRAM loading
    and JSON schema grammar compilation never cause the first texture to time out.
    """
    base_url = ollama_url.rstrip("/")
    print(f"[*] Checking Ollama server at {base_url} for model '{model}'...")
    try:
        resp = HTTP_SESSION.get(f"{base_url}/api/tags", timeout=10)
        resp.raise_for_status()
        models_data = resp.json().get("models", [])
    except Exception as exc:
        print(f"[!] ERROR: Could not connect to Ollama at {base_url}: {exc}")
        print("    Make sure 'ollama serve' is running, or pass --no-llm to use heuristics.")
        return False

    available_names: List[str] = [m.get("name", "") for m in models_data if isinstance(m, dict)]
    matched_model = None
    for m in models_data:
        name = m.get("name", "")
        if name == model or name == f"{model}:latest" or name.split(":")[0] == model:
            matched_model = m
            break

    if matched_model is None:
        print(f"[!] ERROR: Model '{model}' was not found in local Ollama.")
        if available_names:
            print(f"    Installed models: {', '.join(available_names)}")
        else:
            print("    No models are currently installed in Ollama.")
        print(f"    Run: ollama pull {model}")
        return False

    details = matched_model.get("details", {})
    param_size = details.get("parameter_size", "unknown size")
    quant = details.get("quantization_level", "unknown quant")
    print(f"[✓] Found model '{matched_model.get('name')}' ({param_size}, {quant}).")
    print(f"[*] Warming up '{model}' in VRAM & compiling JSON schema (may take 15-60s on cold start)...")

    t0 = time.perf_counter()
    try:
        warmup_payload = {
            "model": model,
            "prompt": "Warmup stone profile. Return JSON.",
            "format": OLLAMA_SCHEMA,
            "stream": False,
            "options": OLLAMA_OPTIONS,
        }
        w_resp = HTTP_SESSION.post(f"{base_url}/api/generate", json=warmup_payload, timeout=180)
        w_resp.raise_for_status()
        dt = time.perf_counter() - t0
        print(f"[✓] Ollama vision pipeline ready in {dt:.1f}s!")
        return True
    except Exception as exc:
        print(f"[!] ERROR: Ollama warm-up failed ({exc}).")
        return False


def heuristic_profile(block_name: str, subdir_hint: str = "") -> Dict[str, Any]:
    """
    Deterministic path/filename heuristics used as fallback or baseline across
    blocks, items, tools, weapons, armor, mobs/creatures/animals/villagers, paintings, and banners.
    """
    name = block_name.lower().replace("\\", "/")
    sub = subdir_hint.lower().replace("\\", "/")

    # Default stone/matte baseline
    profile: Dict[str, Any] = {
        "category": "stone",
        "normal_strength": 2.0,
        "height_contrast": 0.22,
        "invert_height": False,
        "base_smoothness": 45,
        "base_f0": 10,
        "porosity_or_sss": 35,
        "mask_mode": "none",
        "mask_threshold": 0.35,
        "accent_smoothness": 45,
        "accent_f0": 10,
        "accent_emissive": 0,
        "accent_height_offset": 0.0,
        "source": "heuristic",
    }

    # 1. Paintings
    if "painting" in sub:
        if name == "back" or name.endswith("/back"):
            profile.update({
                "category": "wood",
                "normal_strength": 1.6,
                "height_contrast": 0.16,
                "base_smoothness": 85,
                "base_f0": 6,
                "porosity_or_sss": 40,
                "source": "builtin:painting_back",
            })
        else:
            profile.update({
                "category": "painting",
                "normal_strength": 2.0,
                "height_contrast": 0.16,
                "base_smoothness": 115,  # Satin oil canvas finish
                "base_f0": 11,           # Oil/varnish dielectric F0
                "porosity_or_sss": 12,
                "accent_smoothness": 165,  # Polished wood frame
                "accent_f0": 6,            # Wood dielectric F0
                "source": "builtin:painting",
            })
        return profile

    # 2. Banners & Shield Pattern Overlays
    if "banner" in sub or name.startswith("banner/") or (name.startswith("shield/") and not name.endswith("/base")):
        is_shield_pat = name.startswith("shield/")
        profile.update({
            "category": "banner",
            "normal_strength": 0.65,
            "height_contrast": 0.08,
            "base_smoothness": 95 if is_shield_pat else 28,
            "base_f0": 8 if is_shield_pat else 4,
            "porosity_or_sss": 30 if is_shield_pat else 62,
            "source": "builtin:banner",
        })
        return profile

    # 3. Mobs, Creatures, Animals, Villagers & Entity Props
    is_entity_folder = "entity" in sub and not name.startswith("equipment/")
    if is_entity_folder:
        # 3a. Pure Emissive Eye Layers, Bioluminescent Overlays & Energy Beams
        if any(
            k in name
            for k in (
                "_eyes", "bioluminescent", "warden_heart", "pulsating_spots",
                "beacon_beam", "gateway_beam", "crystal_beam", "end_portal", "experience_orb",
            )
        ):
            profile.update({
                "category": "emissive",
                "normal_strength": 0.45,
                "height_contrast": 0.05,
                "base_smoothness": 200,
                "base_f0": 20,
                "porosity_or_sss": 0,
                "mask_mode": "bright_pixels",
                "mask_threshold": 0.05,
                "accent_smoothness": 215,
                "accent_f0": 24,
                "accent_emissive": 245,
                "source": "builtin:entity_emissive",
            })
            return profile

        # 3b. Villager & Zombie Villager Profession Level Badges (Stone, Iron, Gold, Emerald, Diamond)
        if "profession_level/" in name:
            if "diamond" in name:
                f0, sm = 28, 235
            elif "emerald" in name:
                f0, sm = 26, 235
            elif "gold" in name:
                f0, sm = 231, 230
            elif "iron" in name:
                f0, sm = 230, 220
            else:
                f0, sm = 10, 65
            profile.update({
                "category": "metal" if f0 >= 230 else "gem",
                "normal_strength": 1.2,
                "height_contrast": 0.14,
                "base_smoothness": sm,
                "base_f0": f0,
                "porosity_or_sss": 0,
                "source": "builtin:villager_badge",
            })
            return profile

        # 3c. Villager & Zombie Villager Professions
        if "profession/" in name:
            if any(k in name for k in ("armorer", "weaponsmith", "toolsmith")):
                profile.update({
                    "category": "mob",
                    "normal_strength": 1.35,
                    "height_contrast": 0.15,
                    "base_smoothness": 62,
                    "base_f0": 7,
                    "porosity_or_sss": 42,
                    "mask_mode": "bright_pixels",
                    "mask_threshold": 0.58,
                    "accent_smoothness": 210,
                    "accent_f0": 230,  # Iron chainmail / plate / buckle accents
                    "accent_height_offset": 0.05,
                })
            elif any(k in name for k in ("cleric", "librarian", "cartographer")):
                profile.update({
                    "category": "mob",
                    "normal_strength": 1.2,
                    "height_contrast": 0.13,
                    "base_smoothness": 38,
                    "base_f0": 5,
                    "porosity_or_sss": 55,
                    "mask_mode": "high_saturation",
                    "mask_threshold": 0.62,
                    "accent_smoothness": 215,
                    "accent_f0": 231,  # Gold trim / monocle / clasp accents
                    "accent_height_offset": 0.04,
                })
            elif any(k in name for k in ("butcher", "leatherworker", "mason", "fletcher", "fisherman")):
                profile.update({
                    "category": "mob",
                    "normal_strength": 1.3,
                    "height_contrast": 0.14,
                    "base_smoothness": 78,  # Supple leather apron
                    "base_f0": 8,
                    "porosity_or_sss": 38,
                })
            else:
                profile.update({
                    "category": "mob",
                    "normal_strength": 1.25,
                    "height_contrast": 0.13,
                    "base_smoothness": 32,  # Woven cloth / straw hat
                    "base_f0": 4,
                    "porosity_or_sss": 60,
                })
            return profile

        # 3d. Base Villagers, Zombie Villagers, Wandering Trader, Illagers & Witch
        if any(k in name for k in ("villager", "wandering_trader", "illager/", "witch")):
            is_ravager = "ravager" in name
            is_vex = "vex" in name
            if is_vex:
                profile.update({
                    "category": "emissive",
                    "normal_strength": 0.85,
                    "height_contrast": 0.10,
                    "base_smoothness": 140,
                    "base_f0": 14,
                    "porosity_or_sss": 180,
                    "mask_mode": "bright_pixels",
                    "mask_threshold": 0.35 if "charging" in name else 0.55,
                    "accent_smoothness": 195,
                    "accent_f0": 20,
                    "accent_emissive": 225,
                })
            elif is_ravager:
                profile.update({
                    "category": "mob",
                    "normal_strength": 1.65,
                    "height_contrast": 0.18,
                    "base_smoothness": 65,
                    "base_f0": 8,
                    "porosity_or_sss": 35,
                    "mask_mode": "bright_pixels",
                    "mask_threshold": 0.55,
                    "accent_smoothness": 195,
                    "accent_f0": 230,  # Iron saddle/chains/horns
                })
            else:
                profile.update({
                    "category": "mob",
                    "normal_strength": 1.15,
                    "height_contrast": 0.12,
                    "base_smoothness": 36,
                    "base_f0": 5,
                    "porosity_or_sss": 72,  # Subtle skin/cloth subsurface scattering
                })
            return profile

        # 3e. Golems & Mechanical Entities (Iron Golem, Copper Golem, Snow Golem, Minecart, Bell, Trident)
        if "iron_golem" in name:
            profile.update({
                "category": "metal",
                "normal_strength": 1.75,
                "height_contrast": 0.18,
                "base_smoothness": 175,
                "base_f0": 230,  # Forged Iron body
                "porosity_or_sss": 0,
                "mask_mode": "high_saturation",
                "mask_threshold": 0.32,
                "accent_smoothness": 35,
                "accent_f0": 4,  # Matte green moss/vines on chest & shoulder
                "accent_height_offset": 0.05,
            })
            return profile

        if "copper_golem" in name:
            oxidized = any(k in name for k in ("oxidized", "weathered", "exposed"))
            profile.update({
                "category": "metal",
                "normal_strength": 1.5,
                "height_contrast": 0.16,
                "base_smoothness": 110 if oxidized else 210,
                "base_f0": 234,  # LabPBR Copper
                "porosity_or_sss": 20 if oxidized else 0,
            })
            return profile

        if "snow_golem" in name:
            profile.update({
                "category": "mob",
                "normal_strength": 1.4,
                "height_contrast": 0.15,
                "base_smoothness": 35,
                "base_f0": 5,
                "porosity_or_sss": 195,  # Strong snow SSS
            })
            return profile

        if any(k in name for k in ("minecart", "trident", "bell/bell_body")):
            f0 = 231 if "bell" in name else 230
            profile.update({
                "category": "metal",
                "normal_strength": 1.35,
                "height_contrast": 0.14,
                "base_smoothness": 215,
                "base_f0": f0,
                "porosity_or_sss": 0,
            })
            return profile

        # 3f. Glowing / Elemental Creatures (Blaze, Allay, Breeze, Magma Cube / Sulfur Cube, Glow Squid, End Crystal)
        if any(k in name for k in ("blaze", "allay", "breeze", "magmacube", "sulfur_cube", "glow_squid", "end_crystal")):
            is_magma_cube = "magmacube" in name or "sulfur_cube" in name
            profile.update({
                "category": "emissive",
                "normal_strength": 1.6 if is_magma_cube else 0.9,
                "height_contrast": 0.18 if is_magma_cube else 0.10,
                "base_smoothness": 65 if is_magma_cube else 185,
                "base_f0": 12 if is_magma_cube else 14,
                "porosity_or_sss": 30 if is_magma_cube else 180,
                "mask_mode": "bright_pixels",
                "mask_threshold": 0.36 if is_magma_cube else 0.22,
                "accent_smoothness": 195,
                "accent_f0": 20,
                "accent_emissive": 235,
                "accent_height_offset": -0.05 if is_magma_cube else 0.04,
            })
            return profile

        # 3g. Aquatic, Amphibious & Slimy Creatures (Fish, Squid, Axolotl, Frog, Tadpole, Dolphin, Turtle, Guardian, Nautilus, Slime)
        if any(
            k in name
            for k in ("slime", "fish/", "squid/", "axolotl", "frog", "tadpole", "dolphin", "turtle", "guardian", "nautilus")
        ):
            is_slime = "slime" in name
            profile.update({
                "category": "mob",
                "normal_strength": 0.75 if is_slime else 1.15,
                "height_contrast": 0.08 if is_slime else 0.12,
                "base_smoothness": 215 if is_slime else 182,  # Wet/mucous sheen
                "base_f0": 14,                                # Water/wet skin dielectric
                "porosity_or_sss": 225 if is_slime else 155,  # Translucent gel / aquatic SSS
            })
            return profile

        # 3h. Furry, Woolly & Feathered Animals (Sheep, Wolf, Cat, Fox, Bear, Panda, Rabbit, Llama, Goat, Camel, Chicken, Parrot, Bat, Bee)
        if any(
            k in name
            for k in (
                "sheep", "wolf", "cat/", "fox", "bear", "panda", "rabbit",
                "llama", "goat", "camel", "chicken", "parrot", "bat", "bee/",
            )
        ):
            if "wolf_armor" in name:
                profile.update({
                    "category": "mob",
                    "normal_strength": 1.5,
                    "height_contrast": 0.16,
                    "base_smoothness": 125,
                    "base_f0": 12,
                    "porosity_or_sss": 25,
                })
            else:
                profile.update({
                    "category": "mob",
                    "normal_strength": 1.25,
                    "height_contrast": 0.13,
                    "base_smoothness": 28,  # Matte fur, wool, or feathers
                    "base_f0": 4,
                    "porosity_or_sss": 58,  # High surface porosity for soft light rolloff
                })
            return profile

        # 3i. Chitinous Arthropods, Hard Shells & Scaled Reptiles (Spider, Endermite, Silverfish, Ender Dragon, Phantom, Shulker, Armadillo)
        if any(
            k in name
            for k in ("spider", "endermite", "silverfish", "enderdragon", "phantom", "shulker", "armadillo")
        ):
            profile.update({
                "category": "mob",
                "normal_strength": 1.45,
                "height_contrast": 0.15,
                "base_smoothness": 132,  # Glossy chitin / keratin shell / dragon scales
                "base_f0": 12,
                "porosity_or_sss": 22,
            })
            return profile

        # 3j. Skeletons & Bone Entities (Skeleton, Stray, Bogged, Wither Skeleton, Parched, Wither)
        if any(k in name for k in ("skeleton", "stray", "bogged", "parched", "wither")):
            profile.update({
                "category": "mob",
                "normal_strength": 1.55,
                "height_contrast": 0.16,
                "base_smoothness": 85,  # Hard satin bone
                "base_f0": 10,
                "porosity_or_sss": 25,
            })
            return profile

        # 3k. Wooden / Ceramic / Fabric Entity Props (Chests, Boats, Shield Base, Armor Stand, Decorated Pot, Cushions)
        if "shield/base" in name or name == "shield_base" or name == "shield_base_nopattern":
            profile.update({
                "category": "wood",
                "normal_strength": 1.5,
                "height_contrast": 0.16,
                "base_smoothness": 85,
                "base_f0": 6,
                "porosity_or_sss": 40,
                "mask_mode": "color_outlier",
                "mask_threshold": 0.25,
                "accent_smoothness": 210,
                "accent_f0": 230,  # Iron rim & studs
                "accent_height_offset": 0.06,
            })
            return profile

        if "chest/" in name and "chest_boat" not in name:
            if "copper" in name:
                oxidized = any(k in name for k in ("oxidized", "weathered", "exposed"))
                profile.update({
                    "category": "metal",
                    "normal_strength": 1.5,
                    "height_contrast": 0.16,
                    "base_smoothness": 115 if oxidized else 210,
                    "base_f0": 234,
                    "porosity_or_sss": 15 if oxidized else 0,
                })
            elif "ender" in name:
                profile.update({
                    "category": "gem",
                    "normal_strength": 1.6,
                    "height_contrast": 0.18,
                    "base_smoothness": 205,
                    "base_f0": 24,
                    "porosity_or_sss": 0,
                })
            else:
                profile.update({
                    "category": "wood",
                    "normal_strength": 1.6,
                    "height_contrast": 0.16,
                    "base_smoothness": 85,
                    "base_f0": 6,
                    "porosity_or_sss": 42,
                })
            return profile

        if any(k in name for k in ("boat", "armorstand", "fishing", "arrow", "creaking", "signs/")):
            profile.update({
                "category": "wood",
                "normal_strength": 1.75 if "creaking" in name else 1.5,
                "height_contrast": 0.18 if "creaking" in name else 0.15,
                "base_smoothness": 75,
                "base_f0": 6,
                "porosity_or_sss": 45,
            })
            return profile

        if "decorated_pot" in name or "conduit" in name:
            profile.update({
                "category": "stone",
                "normal_strength": 1.5,
                "height_contrast": 0.16,
                "base_smoothness": 95,
                "base_f0": 10,
                "porosity_or_sss": 35,
            })
            return profile

        if "cushion" in name or "lead_knot" in name:
            profile.update({
                "category": "soft",
                "normal_strength": 1.3,
                "height_contrast": 0.13,
                "base_smoothness": 28,
                "base_f0": 4,
                "porosity_or_sss": 60,
            })
            return profile

        # 3l. Remaining Land Animals & Humanoid/Monster Mobs (Cow, Pig, Hoglin, Horse, Strider, Sniffer, Zombie, Piglin, Creeper, Enderman, Warden, Ghast)
        is_drowned = "drowned" in name
        is_piglin = "piglin" in name
        profile.update({
            "category": "mob",
            "normal_strength": 1.35,
            "height_contrast": 0.14,
            "base_smoothness": 135 if is_drowned else 54,
            "base_f0": 12 if is_drowned else 7,
            "porosity_or_sss": 42,
            "mask_mode": "high_saturation" if is_piglin else "none",
            "mask_threshold": 0.58,
            "accent_smoothness": 215 if is_piglin else 54,
            "accent_f0": 231 if is_piglin else 7,  # Golden belt/jewelry accents on Piglins
        })
        return profile

    # 4. Tools, Weapons, Armor, Equipment, and Ingots (Item & 3D Equipment Folders)
    is_tool_or_weapon = any(
        k in name
        for k in ("_sword", "_pickaxe", "_axe", "_shovel", "_hoe", "trident", "mace", "shears", "shield", "flint_and_steel", "brush")
    )
    is_armor_or_metal_item = any(
        k in name
        for k in (
            "_helmet", "_chestplate", "_leggings", "_boots", "_horse_armor", "horse_body/",
            "humanoid/", "humanoid_leggings/", "humanoid_baby/", "nautilus_body/", "wolf_body/",
            "_ingot", "_nugget", "chainmail", "bucket", "compass", "clock", "spyglass", "minecart", "cauldron", "hopper",
        )
    )

    if is_tool_or_weapon or is_armor_or_metal_item:
        if any(k in name for k in ("wooden_", "wood_", "leather", "saddle", "harness", "turtle")):
            is_leather = any(k in name for k in ("leather", "saddle", "harness"))
            profile.update({
                "category": "soft" if is_leather else "wood",
                "normal_strength": 1.3,
                "height_contrast": 0.14,
                "base_smoothness": 95 if is_leather else 75,
                "base_f0": 8 if is_leather else 6,
                "porosity_or_sss": 40,
            })
            return profile

        if "stone_" in name:
            profile.update({
                "category": "tool",
                "normal_strength": 1.8,
                "height_contrast": 0.20,
                "base_smoothness": 55,
                "base_f0": 10,
                "porosity_or_sss": 35,
                "mask_mode": "wood_handle",
                "mask_threshold": 0.25,
                "accent_smoothness": 75,
                "accent_f0": 6,
                "accent_height_offset": -0.04,
            })
            return profile

        if "diamond" in name:
            f0 = 28
            smooth = 235
        elif any(k in name for k in ("gold", "golden", "clock", "bell")):
            f0 = 231
            smooth = 230
        elif any(k in name for k in ("copper", "spyglass", "brush", "lightning")):
            f0 = 234
            smooth = 215
        elif "netherite" in name:
            f0 = 235
            smooth = 195
        else:
            f0 = 230
            smooth = 220

        has_wood_handle = is_tool_or_weapon and not any(k in name for k in ("trident", "shears", "flint_and_steel"))
        profile.update({
            "category": "tool" if has_wood_handle else ("gem" if f0 < 230 else "metal"),
            "normal_strength": 1.25,
            "height_contrast": 0.15,
            "base_smoothness": smooth,
            "base_f0": f0,
            "porosity_or_sss": 0,
            "mask_mode": "wood_handle" if has_wood_handle else "none",
            "mask_threshold": 0.25,
            "accent_smoothness": 75,
            "accent_f0": 6,
            "accent_emissive": 0,
            "accent_height_offset": -0.05 if has_wood_handle else 0.0,
        })
        return profile

    # 5. Ores (Composite stone + metallic, gemstone, or coal veins)
    is_ore_path = (
        "_ore" in name
        or name.startswith("ore/")
        or "/ore/" in name
        or "deepslate_ores/" in name
        or "ancient_debris" in name
        or "gilded_blackstone" in name
        or "raw_iron" in name
        or "raw_gold" in name
        or "raw_copper" in name
    )
    if is_ore_path:
        if "coal" in name:
            profile.update({
                "category": "ore",
                "normal_strength": 2.4,
                "height_contrast": 0.25,
                "base_smoothness": 40,
                "base_f0": 10,
                "mask_mode": "dark_pixels",
                "mask_threshold": 0.62,
                "accent_smoothness": 135,
                "accent_f0": 16,
                "accent_emissive": 0,
                "accent_height_offset": 0.05,
            })
        else:
            is_metal_ore = any(m in name for m in ("iron", "gold", "copper", "gilded", "debris"))
            accent_f0 = (
                231 if "gold" in name or "gilded" in name
                else (234 if "copper" in name else (230 if is_metal_ore else 28))
            )
            is_lit_redstone = "redstone" in name and "redstone_off" not in name
            profile.update({
                "category": "ore",
                "normal_strength": 2.4,
                "height_contrast": 0.25,
                "base_smoothness": 40,
                "base_f0": 10,
                "mask_mode": "color_outlier",
                "mask_threshold": 0.24,
                "accent_smoothness": 215,
                "accent_f0": accent_f0,
                "accent_emissive": 140 if is_lit_redstone else 0,
                "accent_height_offset": 0.08,
            })

    # 6. Solid Metals
    elif any(k in name for k in ("iron_block", "gold_block", "copper", "anvil", "cauldron", "chain", "lantern", "netherite", "rail", "hopper", "iron_door", "iron_trapdoor", "iron_bars", "bell")):
        f0 = 231 if "gold" in name else (234 if "copper" in name else 230)
        oxidized = any(k in name for k in ("oxidized", "weathered", "exposed"))
        profile.update({
            "category": "metal",
            "normal_strength": 1.4,
            "height_contrast": 0.14,
            "base_smoothness": 115 if oxidized else 210,
            "base_f0": f0,
            "porosity_or_sss": 15 if oxidized else 0,
        })

    # 7. Gems, Ice, Glass, Potions, Pearls, Polished Blocks
    elif any(k in name for k in ("diamond", "emerald", "amethyst", "quartz", "ice", "glass", "polished_", "glazed_", "ender_pearl", "ender_eye", "potion", "bottle", "ruby", "sapphire", "prismarine_crystals", "echo_shard")):
        profile.update({
            "category": "gem",
            "normal_strength": 1.1,
            "height_contrast": 0.10,
            "base_smoothness": 225,
            "base_f0": 26,
            "porosity_or_sss": 140 if any(s in name for s in ("ice", "amethyst", "quartz")) else 0,
        })

    # 8. Emissive Blocks & Items
    elif any(k in name for k in ("glowstone", "sea_lantern", "shroomlight", "froglight", "lava", "magma", "fire", "soul_fire", "jack_o_lantern", "beacon", "redstone_lamp_on", "crying_obsidian", "respawn_anchor", "emissive", "blaze", "glow_ink", "glow_berries", "nether_star", "end_rod", "torch")):
        is_cracked = any(k in name for k in ("magma", "glowstone", "crying_obsidian", "respawn_anchor", "jack_o_lantern", "redstone_lamp_on"))
        profile.update({
            "category": "emissive",
            "normal_strength": 1.8 if is_cracked else 0.8,
            "height_contrast": 0.20 if is_cracked else 0.08,
            "base_smoothness": 60,
            "base_f0": 12,
            "mask_mode": "bright_pixels",
            "mask_threshold": 0.42 if is_cracked else 0.15,
            "accent_smoothness": 180,
            "accent_f0": 20,
            "accent_emissive": 230,
            "accent_height_offset": -0.08 if "magma" in name else 0.05,
        })

    # 9. Woods, Planks, Logs, Bows, Boats, Doors, Signs, Fences
    elif any(k in name for k in ("planks", "log", "wood", "stem", "hyphae", "door", "trapdoor", "barrel", "chest", "crafting_table", "bookshelf", "bamboo", "sign", "fence", "gate", "beehive", "ladder", "stick", "bow", "crossbow", "fishing_rod", "boat", "raft", "bowl")):
        profile.update({
            "category": "wood",
            "normal_strength": 1.7,
            "height_contrast": 0.16,
            "base_smoothness": 75,
            "base_f0": 6,
            "porosity_or_sss": 45,
        })

    # 10. Foliage, Plants, Snow, Wool, Beds, Cloth/Food
    elif any(k in name for k in ("leaves", "sapling", "grass", "fern", "vine", "lily", "flower", "tulip", "orchid", "moss", "snow", "wool", "carpet", "beds/", "potted_plants", "dripleaf", "roots", "wheat", "cobweb", "mushroom", "bee_nest", "apple", "bread", "carrot", "potato", "melon", "berries", "chorus", "feather", "string", "paper", "book", "bundle")):
        is_sss = not any(w in name for w in ("wool", "carpet", "beds/", "book", "paper", "bundle"))
        profile.update({
            "category": "foliage" if is_sss else "soft",
            "normal_strength": 1.5,
            "height_contrast": 0.15,
            "base_smoothness": 30,
            "base_f0": 4,
            "porosity_or_sss": 185 if is_sss else 60,
        })

    return profile


def query_ollama_vlm(
    image: Image.Image,
    block_name: str,
    model: str,
    ollama_url: str,
    subdir_hint: str = "",
) -> Optional[Dict[str, Any]]:
    """
    Sends the first square frame of the texture (upscaled via Nearest Neighbor if low-res so
    pixel boundaries remain sharp) plus the heuristic baseline to a local Ollama vision model.
    """
    w, h = image.size
    frame_h = min(w, h) if (h >= w and h % w == 0) else h
    frame = image.crop((0, 0, w, frame_h)).convert("RGBA")

    target_dim = 112 if max(w, frame_h) <= 64 else 224
    upscaled = frame.resize((target_dim, target_dim), resample=Image.Resampling.NEAREST)

    buf = io.BytesIO()
    upscaled.save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    fallback = heuristic_profile(block_name, subdir_hint=subdir_hint)
    baseline_json = json.dumps({k: v for k, v in fallback.items() if k != "source"})

    prompt = (
        f"Minecraft {w}x{frame_h} texture: '{block_name}' (folder: '{subdir_hint or 'block'}').\n"
        f"Baseline LabPBR 1.3 profile: {baseline_json}\n"
        "Inspect the texture image and refine the JSON profile:\n"
        "- Keep normal_strength between 0.8 (smooth skin/fur) and 3.2 (deep relief/bricks).\n"
        "- LabPBR F0 codes: 4-15 for matte dielectrics (wood/stone/soil/cloth/fur/skin), 20-35 for gems/glass/ice, "
        "230=Iron, 231=Gold, 232=Aluminum, 233=Chrome, 234=Copper, 235=Lead/Netherite, 236=Platinum, 237=Silver, 255=Tinted Metal.\n"
        "- For mobs, animals, and villagers, keep category='mob' and use natural dielectric F0 (4-14) unless it is a metal golem (Iron=230, Copper=234).\n"
        "- For metal/diamond swords, pickaxes, axes, shovels, and hoes, set base_f0 to the blade metal/gem (e.g. 230=Iron, 231=Gold, 234=Copper, 28=Diamond), "
        "base_smoothness=215-235, and mask_mode='wood_handle' with accent_f0=6, accent_smoothness=75 for the wooden grip.\n"
        "Return ONLY valid JSON."
    )

    payload = {
        "model": model,
        "prompt": prompt,
        "images": [img_b64],
        "format": OLLAMA_SCHEMA,
        "stream": False,
        "options": OLLAMA_OPTIONS,
    }

    try:
        resp = HTTP_SESSION.post(f"{ollama_url.rstrip('/')}/api/generate", json=payload, timeout=180)
        resp.raise_for_status()
        raw_response = resp.json().get("response", "{}")
        parsed = json.loads(raw_response)
        return sanitize_profile(parsed, fallback=fallback, source=f"llm:{model}")
    except Exception as exc:
        print(f"  [!] Ollama query failed for '{block_name}' ({exc}); falling back to heuristic profile.")
        return None


# =====================================================================
# Pixel-Exact Math & LabPBR Channel Packing
# =====================================================================
def compute_luminance_and_saturation(rgb_float: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Computes perceptual luminance [0, 1] and HSV saturation [0, 1] from RGB float [0, 1]."""
    r, g, b = rgb_float[..., 0], rgb_float[..., 1], rgb_float[..., 2]
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    cmax = np.max(rgb_float, axis=-1)
    cmin = np.min(rgb_float, axis=-1)
    delta = cmax - cmin
    sat = np.where(cmax > 1e-5, delta / np.maximum(cmax, 1e-5), 0.0)
    return lum, sat


def compute_accent_mask(
    rgb_float: np.ndarray,
    alpha_mask: np.ndarray,
    lum: np.ndarray,
    sat: np.ndarray,
    mode: str,
    threshold: float,
    base_f0: int = 10,
) -> np.ndarray:
    """
    Generates a soft [0.0, 1.0] weight mask for composite materials (e.g., ore specks,
    glowing veins, or wooden tool handles) strictly aligned to the original bitmap pixels.
    """
    if mode == "none" or not np.any(alpha_mask):
        return np.zeros_like(lum)

    valid_rgb = rgb_float[alpha_mask]
    ramp = max(min(1.0 - threshold, MASK_RAMP_WIDTH), 0.08)

    if mode == "wood_handle":
        r, g, b = rgb_float[..., 0], rgb_float[..., 1], rgb_float[..., 2]
        h, w = lum.shape
        frame_h = min(w, h) if (h >= w and h % w == 0) else h
        y_coords = (np.arange(h) % frame_h)[:, np.newaxis] / max(frame_h - 1, 1)
        x_coords = np.arange(w)[np.newaxis, :] / max(w - 1, 1)

        is_brown = (r > g * 1.08) & (g > b * 1.12) & (lum > 0.08) & (lum < 0.62) & (b < 0.42)
        if base_f0 in (231, 234):
            in_grip_zone = (y_coords >= 0.42) & (x_coords <= 0.58)
            is_brown = is_brown & (lum < 0.46) & (sat < 0.68) & in_grip_zone

        mask = np.where(is_brown, 1.0, 0.0)

    elif mode == "color_outlier":
        median_rgb = np.median(valid_rgb, axis=0)
        color_dist = np.linalg.norm(rgb_float - median_rgb, axis=-1)

        chroma_vec = rgb_float - lum[..., np.newaxis]
        med_chroma = np.median(chroma_vec[alpha_mask], axis=0)
        chroma_dist = np.linalg.norm(chroma_vec - med_chroma, axis=-1)

        median_sat = float(np.median(sat[alpha_mask]))
        sat_boost = np.clip(sat - median_sat, 0.0, 1.0) * 0.65

        score = color_dist + chroma_dist * 0.85 + sat_boost
        mask = np.clip((score - threshold) / ramp, 0.0, 1.0)

    elif mode == "high_saturation":
        mask = np.clip((sat - threshold) / ramp, 0.0, 1.0)

    elif mode == "bright_pixels":
        mask = np.clip((lum - threshold) / ramp, 0.0, 1.0)

    elif mode == "dark_pixels":
        mask = np.clip(((1.0 - lum) - threshold) / ramp, 0.0, 1.0)

    else:
        mask = np.zeros_like(lum)

    return np.where(alpha_mask, mask, 0.0)


def inpaint_uv_island_margins(height_map: np.ndarray, alpha_mask: np.ndarray) -> np.ndarray:
    """
    For 3D entity box UV sheets (mobs, animals, villagers, 3D armor), replaces transparent
    gutter pixels immediately bordering opaque UV islands with the local average height
    of their opaque neighbors. This prevents false normal seams along 3D cube edges.
    """
    if np.all(alpha_mask) or not np.any(alpha_mask):
        return height_map

    h_pad = np.pad(np.where(alpha_mask, height_map, 0.0), pad_width=1, mode="edge")
    m_pad = np.pad(alpha_mask.astype(np.float32), pad_width=1, mode="edge")

    neighbor_sum = (
        h_pad[:-2, :-2] + h_pad[:-2, 1:-1] + h_pad[:-2, 2:]
        + h_pad[1:-1, :-2] + h_pad[1:-1, 2:]
        + h_pad[2:, :-2] + h_pad[2:, 1:-1] + h_pad[2:, 2:]
    )
    neighbor_cnt = (
        m_pad[:-2, :-2] + m_pad[:-2, 1:-1] + m_pad[:-2, 2:]
        + m_pad[1:-1, :-2] + m_pad[1:-1, 2:]
        + m_pad[2:, :-2] + m_pad[2:, 1:-1] + m_pad[2:, 2:]
    )

    dilated = np.where(neighbor_cnt > 0, neighbor_sum / np.maximum(neighbor_cnt, 1.0), 0.82)
    return np.where(alpha_mask, height_map, dilated)


def generate_frame_normal_and_ao(
    height_frame: np.ndarray,
    normal_strength: float,
    flip_y: bool = False,
    pad_mode: str = "wrap",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes X/Y normals and local cavity Ambient Occlusion for a single frame.
    Uses toroidal padding (mode='wrap') for seamless tiling blocks, or 'edge' padding
    for non-tiling cutout items, entities, paintings, and banners.
    """
    padded = np.pad(height_frame, pad_width=1, mode=pad_mode)

    # 3x3 Scharr operator for crisp pixel-art edge responsiveness at any resolution
    dx = (
        3.0 * (padded[:-2, 2:] - padded[:-2, :-2])
        + 10.0 * (padded[1:-1, 2:] - padded[1:-1, :-2])
        + 3.0 * (padded[2:, 2:] - padded[2:, :-2])
    ) / 32.0

    dy = (
        3.0 * (padded[2:, :-2] - padded[:-2, :-2])
        + 10.0 * (padded[2:, 1:-1] - padded[:-2, 1:-1])
        + 3.0 * (padded[2:, 2:] - padded[:-2, 2:])
    ) / 32.0

    y_sign = -1.0 if flip_y else 1.0

    nx = -dx * normal_strength * 4.0
    ny = y_sign * dy * normal_strength * 4.0
    nz = np.ones_like(height_frame)

    norm = np.sqrt(nx * nx + ny * ny + nz * nz)
    nx /= norm
    ny /= norm

    normal_r = np.clip(np.round((nx * 0.5 + 0.5) * 255.0), 0, 255).astype(np.uint8)
    normal_g = np.clip(np.round((ny * 0.5 + 0.5) * 255.0), 0, 255).astype(np.uint8)

    # Local 3x3 neighborhood average for cavity Ambient Occlusion (LabPBR Blue channel)
    local_mean = (
        padded[:-2, :-2] + padded[:-2, 1:-1] + padded[:-2, 2:]
        + padded[1:-1, :-2] + padded[1:-1, 1:-1] + padded[1:-1, 2:]
        + padded[2:, :-2] + padded[2:, 1:-1] + padded[2:, 2:]
    ) / 9.0

    cavity = np.clip(local_mean - height_frame, 0.0, 1.0)
    ao = np.clip(np.round((1.0 - cavity * 1.6) * 255.0), 140, 255).astype(np.uint8)

    return normal_r, normal_g, ao


def build_painting_labpbr_maps(
    img: Image.Image,
    flip_y: bool = False,
    pack_border_width: Optional[int] = None,
) -> Tuple[Image.Image, Image.Image]:
    """
    Specialized LabPBR builder for Minecraft paintings:
      - Automatically detects the outer wooden frame border thickness (in pixels).
      - Frame border receives 3D beveled Scharr normals + Polished Wood specular (smoothness ~165, F0=6).
      - Interior canvas receives a 100% flat normal map (128, 128) + Satin Oil Painting finish (smoothness ~115, F0=11).
    """
    rgba = np.array(img.convert("RGBA"), dtype=np.uint8)
    h, w, _ = rgba.shape
    rgb_float = rgba[..., :3].astype(np.float32) / 255.0
    alpha_mask = rgba[..., 3] > 16
    lum, _ = compute_luminance_and_saturation(rgb_float)

    border_w = detect_single_painting_border(rgb_float, pack_border=pack_border_width)

    yy, xx = np.ogrid[:h, :w]
    dist_to_edge = np.minimum(np.minimum(yy, h - 1 - yy), np.minimum(xx, w - 1 - xx))
    frame_mask = (dist_to_edge < border_w) & alpha_mask
    canvas_mask = (dist_to_edge >= border_w) & alpha_mask

    frame_med = float(np.median(lum[frame_mask])) if np.any(frame_mask) else 0.5
    canvas_med = float(np.median(lum[canvas_mask])) if np.any(canvas_mask) else 0.5

    height_float = np.where(
        frame_mask,
        np.clip(0.90 + (lum - frame_med) * 0.18, 0.80, 1.0),
        0.78,
    )
    height_float = np.where(alpha_mask, height_float, 1.0)
    height_u8 = np.clip(np.round(height_float * 254.0 + 1.0), 1, 255).astype(np.uint8)

    nr, ng, ao = generate_frame_normal_and_ao(height_float, normal_strength=2.0, flip_y=flip_y, pad_mode="edge")
    inner_flat = dist_to_edge > border_w
    nr = np.where(inner_flat | ~alpha_mask, 128, nr).astype(np.uint8)
    ng = np.where(inner_flat | ~alpha_mask, 128, ng).astype(np.uint8)
    ao = np.where(inner_flat | ~alpha_mask, 255, ao).astype(np.uint8)
    height_u8 = np.where(alpha_mask, height_u8, 255).astype(np.uint8)

    n_rgba = np.stack([nr, ng, ao, height_u8], axis=-1)

    frame_smooth = np.clip(np.round(165.0 + (lum - frame_med) * 24.0), 120, 210)
    canvas_smooth = np.clip(np.round(115.0 + (lum - canvas_med) * 14.0), 95, 138)

    spec_r = np.where(frame_mask, frame_smooth, canvas_smooth).astype(np.uint8)
    spec_g = np.where(frame_mask, 6, 11).astype(np.uint8)
    spec_b = np.where(frame_mask, 20, 12).astype(np.uint8)
    spec_a = np.zeros((h, w), dtype=np.uint8)

    spec_r = np.where(alpha_mask, spec_r, 0)
    spec_g = np.where(alpha_mask, spec_g, 0)
    spec_b = np.where(alpha_mask, spec_b, 0)

    s_rgba = np.stack([spec_r, spec_g, spec_b, spec_a], axis=-1)
    return Image.fromarray(n_rgba, mode="RGBA"), Image.fromarray(s_rgba, mode="RGBA")


def build_banner_labpbr_maps(
    img: Image.Image,
    block_name: str,
    profile: Dict[str, Any],
    flip_y: bool = False,
) -> Tuple[Image.Image, Image.Image]:
    """
    Specialized LabPBR builder for Minecraft banners and shield pattern overlays:
      - Extracts only clamped high-pass micro-weave (ignoring macro pattern edges/alpha transitions)
        so high-res woven cloth textures retain subtle fabric grain while flat low-res patterns stay uniform.
      - On banner_base.png, separates the top cloth flag UV region (y < 40/64*H) from the bottom
        wooden crossbar/stand pole UV region (y >= 40/64*H).
    """
    rgba = np.array(img.convert("RGBA"), dtype=np.uint8)
    h, w, _ = rgba.shape
    rgb_float = rgba[..., :3].astype(np.float32) / 255.0
    alpha = rgba[..., 3]
    alpha_mask = alpha > 16
    lum, _ = compute_luminance_and_saturation(rgb_float)

    is_banner_base = "banner_base" in block_name.lower()
    yy = np.arange(h)[:, np.newaxis]
    wood_uv_mask = (yy >= int(h * 40 / 64)) & alpha_mask if is_banner_base else np.zeros_like(alpha_mask)
    cloth_mask = alpha_mask & ~wood_uv_mask

    padded_lum = np.pad(lum, pad_width=1, mode="edge")
    local_mean = (
        padded_lum[:-2, :-2] + padded_lum[:-2, 1:-1] + padded_lum[:-2, 2:]
        + padded_lum[1:-1, :-2] + padded_lum[1:-1, 1:-1] + padded_lum[1:-1, 2:]
        + padded_lum[2:, :-2] + padded_lum[2:, 1:-1] + padded_lum[2:, 2:]
    ) / 9.0

    padded_a = np.pad(alpha, pad_width=1, mode="edge")
    min_neighbor_a = np.minimum.reduce([
        padded_a[:-2, 1:-1], padded_a[2:, 1:-1], padded_a[1:-1, :-2], padded_a[1:-1, 2:], alpha
    ])
    interior_opaque = min_neighbor_a >= 240

    cloth_micro = np.where(interior_opaque, np.clip(lum - local_mean, -0.035, 0.035), 0.0)
    wood_med = float(np.median(lum[wood_uv_mask])) if np.any(wood_uv_mask) else 0.5

    height_float = np.where(
        wood_uv_mask,
        np.clip(0.84 + (lum - wood_med) * 0.16, 0.70, 1.0),
        0.82 + cloth_micro * 0.8,
    )
    height_float = inpaint_uv_island_margins(height_float, alpha_mask)
    height_u8 = np.clip(np.round(np.where(alpha_mask, height_float, 1.0) * 254.0 + 1.0), 1, 255).astype(np.uint8)

    nr, ng, ao = generate_frame_normal_and_ao(height_float, normal_strength=0.75, flip_y=flip_y, pad_mode="edge")
    nr = np.where(alpha_mask, nr, 128).astype(np.uint8)
    ng = np.where(alpha_mask, ng, 128).astype(np.uint8)
    ao = np.where(alpha_mask, np.where(cloth_mask, np.maximum(ao, 235), ao), 255).astype(np.uint8)
    height_u8 = np.where(alpha_mask, height_u8, 255).astype(np.uint8)

    n_rgba = np.stack([nr, ng, ao, height_u8], axis=-1)

    base_sm = int(profile.get("base_smoothness", 28))
    base_f0 = int(profile.get("base_f0", 4))
    base_por = int(profile.get("porosity_or_sss", 62))

    spec_r = np.where(wood_uv_mask, 80, base_sm).astype(np.uint8)
    spec_g = np.where(wood_uv_mask, 6, base_f0).astype(np.uint8)
    spec_b = np.where(wood_uv_mask, 45, base_por).astype(np.uint8)
    spec_a = np.zeros((h, w), dtype=np.uint8)

    spec_r = np.where(alpha_mask, spec_r, 0)
    spec_g = np.where(alpha_mask, spec_g, 0)
    spec_b = np.where(alpha_mask, spec_b, 0)

    s_rgba = np.stack([spec_r, spec_g, spec_b, spec_a], axis=-1)
    return Image.fromarray(n_rgba, mode="RGBA"), Image.fromarray(s_rgba, mode="RGBA")


def build_labpbr_maps(
    img: Image.Image,
    profile: Dict[str, Any],
    flip_y: bool = False,
    emissive_img: Optional[Image.Image] = None,
    block_name: str = "",
    subdir_hint: str = "",
    pack_painting_border: Optional[int] = None,
) -> Tuple[Image.Image, Image.Image]:
    """
    Produces LabPBR 1.3 compliant _n.png and _s.png images matching the exact dimensions
    and pixel grid of the source texture (at any resolution).
    Dispatches paintings and banners to their specialized builders, inpaints 3D entity
    UV margins to prevent box-edge seams, and bevels 2D held item cutouts.
    """
    profile = sanitize_profile(profile)
    category = profile["category"]

    if category == "painting":
        return build_painting_labpbr_maps(img, flip_y=flip_y, pack_border_width=pack_painting_border)
    if category == "banner":
        return build_banner_labpbr_maps(img, block_name=block_name, profile=profile, flip_y=flip_y)

    rgba = np.array(img.convert("RGBA"), dtype=np.uint8)
    h, w, _ = rgba.shape

    rgb_float = rgba[..., :3].astype(np.float32) / 255.0
    alpha = rgba[..., 3]
    alpha_mask = alpha > 16  # Ignore transparent background pixels
    has_transparency = not np.all(alpha_mask)
    is_entity_uv = "entity" in subdir_hint.lower() or "armor" in subdir_hint.lower() or category == "mob"

    lum, sat = compute_luminance_and_saturation(rgb_float)
    base_f0 = sanitize_f0(profile["base_f0"], 10)

    accent_mask = compute_accent_mask(
        rgb_float=rgb_float,
        alpha_mask=alpha_mask,
        lum=lum,
        sat=sat,
        mode=str(profile["mask_mode"]),
        threshold=float(profile["mask_threshold"]),
        base_f0=base_f0,
    )

    # 1. Construct Height Map (Normal Alpha Channel)
    base_h = (1.0 - lum) if profile["invert_height"] else lum
    if np.any(alpha_mask):
        med_h = float(np.median(base_h[alpha_mask]))
    else:
        med_h = 0.5

    contrast = float(profile["height_contrast"])
    accent_offset = float(profile["accent_height_offset"])

    height_float = 0.82 + (base_h - med_h) * contrast + (accent_mask * accent_offset)
    height_float = np.clip(height_float, 0.05, 1.0)

    if is_entity_uv and has_transparency:
        # Inpaint 1px UV island margins so 3D mob/villager/armor box corners have zero seam step
        height_for_normals = inpaint_uv_island_margins(height_float, alpha_mask)
        pad_mode = "edge"
    elif has_transparency:
        # For 2D held item cutouts (swords, axes, ingots), bevel outer silhouette outward
        height_for_normals = np.where(alpha_mask, height_float, 0.74)
        pad_mode = "edge"
    else:
        height_for_normals = height_float
        pad_mode = "wrap"

    # LabPBR spec: Height must be [1, 255]; 0 is invalid and breaks POM in many shaders
    height_u8 = np.clip(np.round(np.where(alpha_mask, height_float, 1.0) * 254.0 + 1.0), 1, 255).astype(np.uint8)

    # 2. Construct Normal (R, G) and AO (B) frame-by-frame (supports animated strips of any resolution!)
    normal_r = np.full((h, w), 128, dtype=np.uint8)
    normal_g = np.full((h, w), 128, dtype=np.uint8)
    ao_b = np.full((h, w), 255, dtype=np.uint8)

    normal_strength = float(profile["normal_strength"])

    if not is_entity_uv and h >= w and h % w == 0:
        num_frames = h // w
        for i in range(num_frames):
            y0, y1 = i * w, (i + 1) * w
            nr, ng, ao = generate_frame_normal_and_ao(
                height_for_normals[y0:y1, :], normal_strength, flip_y=flip_y, pad_mode=pad_mode
            )
            normal_r[y0:y1, :] = nr
            normal_g[y0:y1, :] = ng
            ao_b[y0:y1, :] = ao
    else:
        normal_r, normal_g, ao_b = generate_frame_normal_and_ao(
            height_for_normals, normal_strength, flip_y=flip_y, pad_mode=pad_mode
        )

    # Neutralize transparent pixels in _n.png
    normal_r = np.where(alpha_mask, normal_r, 128)
    normal_g = np.where(alpha_mask, normal_g, 128)
    ao_b = np.where(alpha_mask, ao_b, 255)
    height_u8 = np.where(alpha_mask, height_u8, 255)

    n_rgba = np.stack([normal_r, normal_g, ao_b, height_u8], axis=-1)

    # 3. Construct Specular Map (_s.png)
    # Red: Perceptual Smoothness (with subtle per-pixel micro-variation tied to luminance)
    base_smooth = float(profile["base_smoothness"])
    accent_smooth = float(profile["accent_smoothness"])
    micro_scale = 16.0 if category == "mob" else 28.0
    micro_var = (lum - med_h) * micro_scale
    smooth_blended = (base_smooth + micro_var) * (1.0 - accent_mask) + (accent_smooth + micro_var * 0.5) * accent_mask
    spec_r = np.clip(np.round(smooth_blended), 0, 255).astype(np.uint8)

    # Green: F0 / Metalness (Must use step threshold, never linear blend, because 230+ are discrete metal IDs!)
    accent_f0 = sanitize_f0(profile["accent_f0"], base_f0)
    is_accent = accent_mask >= ACCENT_STEP_CUTOFF
    spec_g = np.where(is_accent, accent_f0, base_f0).astype(np.uint8)

    # Blue: Porosity (0-64) or Subsurface Scattering (65-255)
    porosity_sss = int(np.clip(profile["porosity_or_sss"], 0, 255))
    accent_porosity = 45 if profile["mask_mode"] == "wood_handle" else (55 if accent_f0 == 4 else 0)
    spec_b = np.where(is_accent, accent_porosity, porosity_sss).astype(np.uint8)

    # Alpha: Emissive (0 = off, 1-254 = emission strength; NEVER 255)
    if emissive_img is not None and emissive_img.size == img.size:
        e_rgba = np.array(emissive_img.convert("RGBA"), dtype=np.float32) / 255.0
        e_lum, _ = compute_luminance_and_saturation(e_rgba[..., :3])
        e_peak = np.max(e_rgba[..., :3], axis=-1)
        e_strength = np.maximum(e_lum, e_peak) * e_rgba[..., 3]
        spec_a = np.where(e_strength > 0.02, np.clip(np.round(e_strength * 253.0 + 1.0), 1, 254), 0).astype(np.uint8)
    else:
        max_emissive = int(np.clip(profile["accent_emissive"], 0, 254))
        if max_emissive > 0:
            rgb_peak = np.max(rgb_float, axis=-1)
            if str(profile.get("source", "")) == "builtin:entity_emissive":
                overlay_glow = np.where(alpha_mask & (rgb_peak > 0.04), np.clip(rgb_peak * 1.15, 0.35, 1.0), 0.0)
                emissive_scaled = np.round(overlay_glow * max_emissive)
            else:
                emissive_scaled = np.round(accent_mask * max_emissive * np.clip( np.maximum(lum, rgb_peak) + 0.25, 0.0, 1.0))
            spec_a = np.clip(emissive_scaled, 0, 254).astype(np.uint8)
        else:
            spec_a = np.zeros((h, w), dtype=np.uint8)

    # Zero out transparent pixels in _s.png to prevent mipmap halo artifacts
    spec_r = np.where(alpha_mask, spec_r, 0)
    spec_g = np.where(alpha_mask, spec_g, 0)
    spec_b = np.where(alpha_mask, spec_b, 0)
    spec_a = np.where(alpha_mask, spec_a, 0)

    s_rgba = np.stack([spec_r, spec_g, spec_b, spec_a], axis=-1)

    return Image.fromarray(n_rgba, mode="RGBA"), Image.fromarray(s_rgba, mode="RGBA")


# =====================================================================
# Pack Metadata & CLI Entrypoint
# =====================================================================
def sync_pack_metadata(
    input_dir: Path,
    output_dir: Path,
    pack_resolution: int = 16,
    default_pack_format: int = 34,
) -> None:
    """
    Mirrors the donor resource pack's pack.mcmeta version/format structure (such as
    Minecraft 26.x's min_format/max_format or legacy pack_format/supported_formats)
    so Minecraft never flags the LabPBR add-on pack as incompatible or outdated.
    Also copies pack.png if present in the donor pack.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    donor_mcmeta_path = input_dir / "pack.mcmeta"
    mcmeta: Dict[str, Any] = {}

    if donor_mcmeta_path.exists():
        try:
            with open(donor_mcmeta_path, "r", encoding="utf-8") as f:
                donor_data = json.load(f)
            if isinstance(donor_data, dict):
                mcmeta = dict(donor_data)
        except Exception as exc:
            print(f"[!] Warning: Could not read donor pack.mcmeta ({exc}); using default metadata.")

    pack_section = dict(mcmeta.get("pack", {})) if isinstance(mcmeta.get("pack"), dict) else {}
    if not any(k in pack_section for k in ("pack_format", "min_format", "max_format", "supported_formats")):
        pack_section["pack_format"] = default_pack_format
        pack_section["supported_formats"] = {"min_inclusive": 15, "max_inclusive": 99}

    donor_desc = pack_section.get("description")
    if isinstance(donor_desc, list):
        pack_section["description"] = list(donor_desc) + [{"text": f" ({pack_resolution}x LabPBR 1.3)", "color": "#55ffff"}]
    else:
        pack_section["description"] = f"Automated {pack_resolution}x LabPBR 1.3 Add-on Pack"

    mcmeta["pack"] = pack_section

    with open(output_dir / "pack.mcmeta", "w", encoding="utf-8") as f:
        json.dump(mcmeta, f, indent=2)

    donor_icon = input_dir / "pack.png"
    if donor_icon.exists():
        shutil.copyfile(donor_icon, output_dir / "pack.png")


def copy_and_sanitize_animation_mcmeta(src_mcmeta: Path, dest_n_mcmeta: Path, dest_s_mcmeta: Path) -> None:
    """
    Copies animation .png.mcmeta files for _n.png and _s.png while stripping non-vanilla
    keys (such as 'random' inside 'animation') that trigger strict codec warnings in modern Minecraft.
    """
    try:
        with open(src_mcmeta, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and isinstance(data.get("animation"), dict):
            clean_anim = {k: v for k, v in data["animation"].items() if k in VALID_ANIMATION_KEYS}
            clean_data = {"animation": clean_anim}
            serialized = json.dumps(clean_data, indent=2)
            dest_n_mcmeta.write_text(serialized, encoding="utf-8")
            dest_s_mcmeta.write_text(serialized, encoding="utf-8")
            return
    except Exception:
        pass

    shutil.copyfile(src_mcmeta, dest_n_mcmeta)
    shutil.copyfile(src_mcmeta, dest_s_mcmeta)


def process_resource_pack(
    input_dir: Path,
    output_dir: Path,
    cache_path: Path,
    use_llm: bool,
    model: str,
    ollama_url: str,
    flip_y: bool,
    refresh_cache: bool = False,
    subdirs: Tuple[str, ...] = DEFAULT_TEXTURE_SUBDIRS,
) -> None:
    pack_resolution = detect_pack_resolution(input_dir)
    print(f"[*] Detected input resource pack base resolution: {pack_resolution}x{pack_resolution}")

    pack_painting_border: Optional[int] = None
    for p_sub in ("assets/minecraft/textures/painting", "assets/minecraft/textures/paintings"):
        p_dir = input_dir / p_sub
        if p_dir.is_dir():
            pack_painting_border = detect_pack_painting_border(p_dir)
            if pack_painting_border is not None:
                print(f"[*] Auto-detected painting wooden frame border width: {pack_painting_border}px")
            break

    if use_llm:
        if not verify_ollama_connection(ollama_url=ollama_url, model=model):
            sys.exit(1)

    profiles_cache: Dict[str, Dict[str, Any]] = {}
    if not refresh_cache and cache_path.exists() and cache_path.stat().st_size > 0:
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                profiles_cache = loaded
                llm_cached = sum(1 for v in profiles_cache.values() if is_llm_profile(v))
                heur_cached = len(profiles_cache) - llm_cached
                print(
                    f"[*] Loaded {len(profiles_cache)} cached profiles from '{cache_path}' "
                    f"({llm_cached} LLM/built-in verified, {heur_cached} heuristic)."
                )
                if use_llm and heur_cached > 0:
                    print(
                        f"[*] LLM mode active: heuristic-only cache entries will be upgraded via Ollama ({model})."
                    )
        except (json.JSONDecodeError, OSError) as exc:
            print(f"[!] Warning: Could not parse '{cache_path}' ({exc}); starting with an empty profile cache.")

    llm_group_cache: Dict[str, Dict[str, Any]] = {}
    for cached_key, cached_prof in profiles_cache.items():
        if is_llm_profile(cached_prof):
            for prefix in subdirs:
                if cached_key.startswith(prefix + "/"):
                    rel_id = cached_key[len(prefix) + 1 :]
                    llm_group_cache[get_material_group_id(prefix, rel_id)] = cached_prof

    sync_pack_metadata(input_dir, output_dir, pack_resolution=pack_resolution)
    processed_count = 0
    llm_cache_hits = 0
    llm_queries_ok = 0
    llm_queries_failed = 0
    heuristic_used = 0
    llm_total_time = 0.0
    cache_dirty = False

    for rel_subdir in subdirs:
        src_folder = input_dir / rel_subdir
        if not src_folder.exists():
            continue

        for root, _, files in os.walk(src_folder):
            root_path = Path(root)
            for file in sorted(files):
                if not file.lower().endswith(".png"):
                    continue
                if file.lower().endswith(("_n.png", "_s.png", "_e.png")):
                    continue

                src_png = root_path / file
                rel_path = src_png.relative_to(input_dir)
                block_key = rel_path.with_suffix("").as_posix()
                file_stem = src_png.stem
                rel_block_id = src_png.relative_to(src_folder).with_suffix("").as_posix()
                group_id = get_material_group_id(rel_subdir, rel_block_id)

                try:
                    img = Image.open(src_png)
                except Exception as exc:
                    print(f"  [!] Skipping unreadable image {rel_path}: {exc}")
                    continue

                emissive_img: Optional[Image.Image] = None
                companion_e = src_png.with_name(f"{file_stem}_e.png")
                if companion_e.exists():
                    try:
                        emissive_img = Image.open(companion_e)
                    except Exception:
                        emissive_img = None

                fallback = heuristic_profile(rel_block_id, subdir_hint=rel_subdir)
                cached_entry = profiles_cache.get(block_key)

                if fallback["category"] in ("painting", "banner") or str(fallback.get("source", "")).startswith("builtin:"):
                    profile = fallback
                    if cached_entry != profile:
                        profiles_cache[block_key] = profile
                        cache_dirty = True
                    heuristic_used += 1
                elif cached_entry is not None and (not use_llm or is_llm_profile(cached_entry)):
                    profile = sanitize_profile(cached_entry, fallback=fallback)
                    if is_llm_profile(profile):
                        llm_cache_hits += 1
                    else:
                        heuristic_used += 1
                elif use_llm and group_id in llm_group_cache:
                    profile = sanitize_profile(llm_group_cache[group_id], fallback=fallback)
                    profiles_cache[block_key] = profile
                    llm_cache_hits += 1
                    cache_dirty = True
                else:
                    profile = None
                    if use_llm:
                        t0 = time.perf_counter()
                        profile = query_ollama_vlm(
                            img, rel_block_id, model=model, ollama_url=ollama_url, subdir_hint=rel_subdir
                        )
                        dt = time.perf_counter() - t0
                        llm_total_time += dt
                        if profile is not None:
                            llm_queries_ok += 1
                            llm_group_cache[group_id] = profile
                            print(
                                f"  [LLM #{llm_queries_ok} | {dt:.2f}s] '{rel_block_id}' -> "
                                f"cat={profile['category']}, smooth={profile['base_smoothness']}, "
                                f"f0={profile['base_f0']}, mask={profile['mask_mode']}"
                                + (
                                    f" (accent_f0={profile['accent_f0']}, em={profile['accent_emissive']})"
                                    if profile["mask_mode"] != "none"
                                    else ""
                                )
                            )
                        else:
                            llm_queries_failed += 1
                    if profile is None:
                        profile = fallback
                        heuristic_used += 1
                    profiles_cache[block_key] = profile
                    cache_dirty = True

                # 2. Generate LabPBR _n.png and _s.png matching exact source sprite dimensions
                n_img, s_img = build_labpbr_maps(
                    img,
                    profile,
                    flip_y=flip_y,
                    emissive_img=emissive_img,
                    block_name=rel_block_id,
                    subdir_hint=rel_subdir,
                    pack_painting_border=pack_painting_border,
                )

                dest_folder = output_dir / rel_path.parent
                dest_folder.mkdir(parents=True, exist_ok=True)

                n_path = dest_folder / f"{file_stem}_n.png"
                s_path = dest_folder / f"{file_stem}_s.png"
                n_img.save(n_path, format="PNG")
                s_img.save(s_path, format="PNG")

                # 3. If the base texture is animated (.png.mcmeta), copy sanitized .mcmeta for _n and _s
                src_mcmeta = src_png.with_name(f"{file}.mcmeta")
                if src_mcmeta.exists():
                    copy_and_sanitize_animation_mcmeta(
                        src_mcmeta,
                        dest_folder / f"{file_stem}_n.png.mcmeta",
                        dest_folder / f"{file_stem}_s.png.mcmeta",
                    )

                processed_count += 1
                if cache_dirty and ((use_llm and llm_queries_ok > 0 and llm_queries_ok % 10 == 0) or processed_count % 50 == 0):
                    with open(cache_path, "w", encoding="utf-8") as f:
                        json.dump(profiles_cache, f, indent=2, sort_keys=True)
                    cache_dirty = False

                if not use_llm and processed_count % 100 == 0:
                    print(f"  [+] Generated {processed_count} LabPBR pairs...")

    if cache_dirty or not cache_path.exists() or cache_path.stat().st_size == 0:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(profiles_cache, f, indent=2, sort_keys=True)

    print(f"\n[✓] Done! Generated {processed_count} LabPBR (_n / _s) texture pairs ({pack_resolution}x pack) in '{output_dir}'.")
    if use_llm:
        avg_t = (llm_total_time / max(llm_queries_ok + llm_queries_failed, 1))
        print(
            f"[✓] LLM Summary ({model}): {llm_queries_ok} new Ollama profiles generated "
            f"(avg {avg_t:.2f}s/tex), {llm_cache_hits} cached/variant-shared LLM profiles reused, "
            f"{llm_queries_failed} failed/fallback."
        )
    else:
        print(f"[✓] Heuristic Mode Summary: {heuristic_used} heuristic/built-in profiles, {llm_cache_hits} cached LLM profiles.")
    print(f"[✓] Material profiles saved to '{cache_path}'.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a LabPBR 1.3 Add-on Pack for any resolution Minecraft Resource Pack.")
    parser.add_argument("-i", "--input", type=Path, required=True, help="Path to extracted base resource pack")
    parser.add_argument("-o", "--output", type=Path, default=Path("build_addon"), help="Output directory for LabPBR add-on")
    parser.add_argument("-c", "--cache", type=Path, default=Path("material_profiles.json"), help="JSON profile cache path")
    parser.add_argument("--no-llm", action="store_true", help="Skip Ollama VLM and use fast deterministic heuristics only")
    parser.add_argument("--refresh-cache", action="store_true", help="Ignore existing material_profiles.json and recompute profiles")
    parser.add_argument("--model", type=str, default="qwen2.5vl:7b", help="Ollama vision model name (e.g., qwen2.5vl:7b, llama3.2-vision)")
    parser.add_argument("--ollama-url", type=str, default="http://localhost:11434", help="Local Ollama server URL")
    parser.add_argument("--flip-y", action="store_true", help="Flip normal map Y channel (DirectX vs OpenGL convention)")
    args = parser.parse_args()

    process_resource_pack(
        input_dir=args.input,
        output_dir=args.output,
        cache_path=args.cache,
        use_llm=not args.no_llm,
        model=args.model,
        ollama_url=args.ollama_url,
        flip_y=args.flip_y,
        refresh_cache=args.refresh_cache,
    )


if __name__ == "__main__":
    main()
