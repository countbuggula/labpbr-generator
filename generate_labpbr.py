#!/usr/bin/env python3
"""
LabPBR 1.3 Automated Add-on Generator for 32x Minecraft Resource Packs.
Preserves original sprites 100% by outputting a standalone overlay resource pack
containing only pack.mcmeta, *_n.png, *_s.png, and animation *.mcmeta files.
"""

import argparse
import base64
import io
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

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

OLLAMA_SCHEMA = {
    "type": "object",
    "properties": {
        "category": {
            "type": "string",
            "enum": ["stone", "wood", "metal", "gem", "ore", "foliage", "emissive", "glass", "soft", "composite"],
        },
        "normal_strength": {"type": "number", "description": "0.5 (flat/smooth) to 4.0 (deep cobble/bricks)"},
        "height_contrast": {"type": "number", "description": "0.05 (flat) to 0.45 (deep relief)"},
        "invert_height": {"type": "boolean", "description": "True if darker pixels are physically raised"},
        "base_smoothness": {"type": "integer", "description": "0 (matte) to 255 (mirror polished)"},
        "base_f0": {"type": "integer", "description": "0-229 dielectric, 230-237 LabPBR metal ID, or 255 tinted metal"},
        "porosity_or_sss": {
            "type": "integer",
            "description": "0-64 for surface porosity (stone/soil/wood), 65-255 for subsurface scattering (leaves/snow/quartz)",
        },
        "mask_mode": {
            "type": "string",
            "enum": ["none", "color_outlier", "high_saturation", "bright_pixels", "dark_pixels"],
            "description": "How to isolate secondary/accent pixels like ore gems or glowing cracks",
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
# Deterministic Fallback Heuristics & Vision LLM Profiler
# =====================================================================
def heuristic_profile(block_name: str) -> Dict[str, Any]:
    """Deterministic filename heuristics used as fallback or baseline."""
    name = block_name.lower()

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
    }

    # Ores (Composite stone + metallic or gemstone veins)
    if "_ore" in name or name in ("ancient_debris_side", "gilded_blackstone"):
        is_metal_ore = any(m in name for m in ("iron", "gold", "copper", "gilded", "debris"))
        accent_f0 = 231 if "gold" in name or "gilded" in name else (234 if "copper" in name else (230 if is_metal_ore else 28))
        profile.update({
            "category": "ore",
            "normal_strength": 2.4,
            "height_contrast": 0.25,
            "base_smoothness": 40,
            "base_f0": 10,
            "mask_mode": "color_outlier",
            "mask_threshold": 0.28,
            "accent_smoothness": 215,
            "accent_f0": accent_f0,
            "accent_emissive": 140 if "redstone" in name else 0,
            "accent_height_offset": 0.08,
        })

    # Solid Metals
    elif any(k in name for k in ("iron_block", "gold_block", "copper", "anvil", "cauldron", "chain", "lantern", "netherite", "rail")):
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

    # Gems, Ice, Glass, Polished Blocks
    elif any(k in name for k in ("diamond_block", "emerald_block", "amethyst", "quartz", "ice", "glass", "polished_", "glazed_")):
        profile.update({
            "category": "gem",
            "normal_strength": 1.1,
            "height_contrast": 0.10,
            "base_smoothness": 225,
            "base_f0": 26,
            "porosity_or_sss": 140 if any(s in name for s in ("ice", "amethyst", "quartz")) else 0,
        })

    # Emissive Blocks
    elif any(k in name for k in ("glowstone", "sea_lantern", "shroomlight", "froglight", "lava", "magma", "fire", "Soul_fire", "jack_o_lantern", "beacon", "redstone_lamp_on")):
        is_cracked = "magma" in name or "glowstone" in name
        profile.update({
            "category": "emissive",
            "normal_strength": 1.8 if is_cracked else 0.8,
            "height_contrast": 0.20 if is_cracked else 0.08,
            "base_smoothness": 60,
            "base_f0": 12,
            "mask_mode": "bright_pixels",
            "mask_threshold": 0.45,
            "accent_smoothness": 180,
            "accent_f0": 20,
            "accent_emissive": 230,
            "accent_height_offset": -0.08 if "magma" in name else 0.05,
        })

    # Woods, Planks, Logs
    elif any(k in name for k in ("planks", "log", "wood", "stem", "hyphae", "door", "trapdoor", "barrel", "chest", "crafting_table", "bookshelf", "bamboo")):
        profile.update({
            "category": "wood",
            "normal_strength": 1.7,
            "height_contrast": 0.16,
            "base_smoothness": 75,
            "base_f0": 6,
            "porosity_or_sss": 45,
        })

    # Foliage, Plants, Snow, Wool
    elif any(k in name for k in ("leaves", "sapling", "grass", "fern", "vine", "lily", "flower", "tulip", "orchid", "moss", "snow", "wool", "carpet")):
        is_sss = not any(w in name for w in ("wool", "carpet"))
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
) -> Optional[Dict[str, Any]]:
    """
    Sends the first 32x32 frame (upscaled 8x via Nearest Neighbor so the vision encoder
    sees sharp pixel boundaries) plus the block filename to a local Ollama vision model.
    """
    w, h = image.size
    frame = image.crop((0, 0, w, min(w, h))).convert("RGBA")
    upscaled = frame.resize((256, 256), resample=Image.Resampling.NEAREST)

    buf = io.BytesIO()
    upscaled.save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    prompt = (
        f"You are generating a LabPBR 1.3 material profile for a 32x32 Minecraft block texture named '{block_name}'.\n"
        "Analyze the block name and pixel art texture to determine its surface properties:\n"
        "- LabPBR F0 codes: 4-15 for matte dielectrics (wood/stone), 20-35 for gems/glass, "
        "230=Iron, 231=Gold, 232=Aluminum, 233=Chrome, 234=Copper, 235=Lead, 236=Platinum, 237=Silver, 255=Tinted Metal.\n"
        "- If the block has embedded secondary materials (like ore gems in stone, glowing cracks in magma, or lit parts), "
        "set mask_mode to 'color_outlier', 'high_saturation', or 'bright_pixels' and configure the accent_* fields.\n"
        "- Never set accent_emissive above 254 (254 is maximum emission in LabPBR; 0 means no emission).\n"
        "Return ONLY valid JSON matching the schema."
    )

    payload = {
        "model": model,
        "prompt": prompt,
        "images": [img_b64],
        "format": OLLAMA_SCHEMA,
        "stream": False,
        "options": {"temperature": 0.1},
    }

    try:
        resp = requests.post(f"{ollama_url.rstrip('/')}/api/generate", json=payload, timeout=45)
        resp.raise_for_status()
        raw_response = resp.json().get("response", "{}")
        return json.loads(raw_response)
    except Exception as exc:
        print(f"  [!] Ollama query failed for '{block_name}' ({exc}); using heuristic profile.")
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
) -> np.ndarray:
    """
    Generates a soft [0.0, 1.0] weight mask for composite materials (e.g., ore specks or glowing veins)
    strictly aligned to the original 32x32 bitmap pixels.
    """
    if mode == "none" or not np.any(alpha_mask):
        return np.zeros_like(lum)

    valid_rgb = rgb_float[alpha_mask]

    if mode == "color_outlier":
        # Median RGB represents the dominant background material (e.g., gray stone in diamond_ore)
        median_rgb = np.median(valid_rgb, axis=0)
        color_dist = np.linalg.norm(rgb_float - median_rgb, axis=-1)
        # Boost distance for pixels with noticeably higher saturation than the background median
        median_sat = float(np.median(sat[alpha_mask]))
        sat_boost = np.clip(sat - median_sat, 0.0, 1.0) * 0.65
        score = color_dist + sat_boost
        mask = np.clip((score - threshold) / max(1.0 - threshold, 0.15), 0.0, 1.0)

    elif mode == "high_saturation":
        mask = np.clip((sat - threshold) / max(1.0 - threshold, 0.15), 0.0, 1.0)

    elif mode == "bright_pixels":
        mask = np.clip((lum - threshold) / max(1.0 - threshold, 0.15), 0.0, 1.0)

    elif mode == "dark_pixels":
        mask = np.clip(((1.0 - lum) - threshold) / max(1.0 - threshold, 0.15), 0.0, 1.0)

    else:
        mask = np.zeros_like(lum)

    return np.where(alpha_mask, mask, 0.0)


def generate_frame_normal_and_ao(
    height_frame: np.ndarray,
    normal_strength: float,
    flip_y: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes seamless wrapped X/Y normals and local cavity Ambient Occlusion for a single square frame.
    Uses toroidal padding (mode='wrap') so 32x32 block edges tile cleanly without seams.
    """
    padded = np.pad(height_frame, pad_width=1, mode="wrap")

    # 3x3 Scharr operator for crisp 32x pixel-art edge responsiveness
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

    # LabPBR standard is OpenGL (+Y up). Invert dy unless DirectX (-Y) is requested.
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


def build_labpbr_maps(
    img: Image.Image,
    profile: Dict[str, Any],
    flip_y: bool = False,
) -> Tuple[Image.Image, Image.Image]:
    """
    Produces LabPBR 1.3 compliant _n.png and _s.png images matching the exact dimensions
    and pixel grid of the source texture.
    """
    rgba = np.array(img.convert("RGBA"), dtype=np.uint8)
    h, w, _ = rgba.shape

    rgb_float = rgba[..., :3].astype(np.float32) / 255.0
    alpha = rgba[..., 3]
    alpha_mask = alpha > 16  # Ignore fully transparent background pixels

    lum, sat = compute_luminance_and_saturation(rgb_float)

    accent_mask = compute_accent_mask(
        rgb_float=rgb_float,
        alpha_mask=alpha_mask,
        lum=lum,
        sat=sat,
        mode=str(profile.get("mask_mode", "none")),
        threshold=float(profile.get("mask_threshold", 0.35)),
    )

    # 1. Construct Height Map (Normal Alpha Channel)
    base_h = (1.0 - lum) if profile.get("invert_height", False) else lum
    # Normalize height around median so contrast scaling stays centered near flush (1.0)
    if np.any(alpha_mask):
        med_h = float(np.median(base_h[alpha_mask]))
    else:
        med_h = 0.5

    contrast = float(profile.get("height_contrast", 0.2))
    accent_offset = float(profile.get("accent_height_offset", 0.0))

    # Height in [0.0, 1.0], biased toward the upper range [0.65, 1.0] so POM doesn't stretch wildly on 32x
    height_float = 0.82 + (base_h - med_h) * contrast + (accent_mask * accent_offset)
    height_float = np.clip(height_float, 0.05, 1.0)
    height_float = np.where(alpha_mask, height_float, 1.0)

    # LabPBR spec: Height must be [1, 255]; 0 is invalid and breaks POM in many shaders
    height_u8 = np.clip(np.round(height_float * 254.0 + 1.0), 1, 255).astype(np.uint8)

    # 2. Construct Normal (R, G) and AO (B) frame-by-frame (supports animated strips!)
    normal_r = np.full((h, w), 128, dtype=np.uint8)
    normal_g = np.full((h, w), 128, dtype=np.uint8)
    ao_b = np.full((h, w), 255, dtype=np.uint8)

    normal_strength = float(profile.get("normal_strength", 1.8))

    if h >= w and h % w == 0:
        num_frames = h // w
        for i in range(num_frames):
            y0, y1 = i * w, (i + 1) * w
            nr, ng, ao = generate_frame_normal_and_ao(height_float[y0:y1, :], normal_strength, flip_y=flip_y)
            normal_r[y0:y1, :] = nr
            normal_g[y0:y1, :] = ng
            ao_b[y0:y1, :] = ao
    else:
        normal_r, normal_g, ao_b = generate_frame_normal_and_ao(height_float, normal_strength, flip_y=flip_y)

    # Neutralize transparent pixels in _n.png
    normal_r = np.where(alpha_mask, normal_r, 128)
    normal_g = np.where(alpha_mask, normal_g, 128)
    ao_b = np.where(alpha_mask, ao_b, 255)
    height_u8 = np.where(alpha_mask, height_u8, 255)

    n_rgba = np.stack([normal_r, normal_g, ao_b, height_u8], axis=-1)

    # 3. Construct Specular Map (_s.png)
    # Red: Perceptual Smoothness (with subtle per-pixel micro-variation tied to luminance)
    base_smooth = float(profile.get("base_smoothness", 45))
    accent_smooth = float(profile.get("accent_smoothness", base_smooth))
    micro_var = (lum - med_h) * 28.0
    smooth_blended = (base_smooth + micro_var) * (1.0 - accent_mask) + (accent_smooth + micro_var * 0.5) * accent_mask
    spec_r = np.clip(np.round(smooth_blended), 0, 255).astype(np.uint8)

    # Green: F0 / Metalness (Must use step threshold, never linear blend, because 230+ are discrete metal IDs!)
    base_f0 = int(np.clip(profile.get("base_f0", 10), 0, 255))
    accent_f0 = int(np.clip(profile.get("accent_f0", base_f0), 0, 255))
    spec_g = np.where(accent_mask >= 0.35, accent_f0, base_f0).astype(np.uint8)

    # Blue: Porosity (0-64) or Subsurface Scattering (65-255)
    porosity_sss = int(np.clip(profile.get("porosity_or_sss", 25), 0, 255))
    # Disable SSS on metallic/gemstone accent pixels in composite blocks
    spec_b = np.where(accent_mask >= 0.35, 0, porosity_sss).astype(np.uint8)

    # Alpha: Emissive (0 = off, 1-254 = emission strength; NEVER 255)
    max_emissive = int(np.clip(profile.get("accent_emissive", 0), 0, 254))
    if max_emissive > 0:
        emissive_scaled = np.round(accent_mask * max_emissive * np.clip(lum + 0.25, 0.0, 1.0))
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
# Pack Scanner & CLI Entrypoint
# =====================================================================
def write_pack_mcmeta(output_dir: Path, pack_format: int = 34) -> None:
    """Writes a clean pack.mcmeta for the generated LabPBR overlay resource pack."""
    mcmeta = {
        "pack": {
            "pack_format": pack_format,
            "supported_formats": {"min_inclusive": 15, "max_inclusive": 64},
            "description": "Automated 32x LabPBR 1.3 Add-on Pack",
        }
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "pack.mcmeta", "w", encoding="utf-8") as f:
        json.dump(mcmeta, f, indent=2)


def process_resource_pack(
    input_dir: Path,
    output_dir: Path,
    cache_path: Path,
    use_llm: bool,
    model: str,
    ollama_url: str,
    flip_y: bool,
    subdirs: Tuple[str, ...] = ("assets/minecraft/textures/block", "assets/minecraft/textures/blocks"),
) -> None:
    # Load cached profiles if present
    profiles_cache: Dict[str, Dict[str, Any]] = {}
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            profiles_cache = json.load(f)
        print(f"[*] Loaded {len(profiles_cache)} cached material profiles from {cache_path}")

    write_pack_mcmeta(output_dir)
    processed_count = 0
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
                # Skip existing PBR maps if the source pack had any stray ones
                if file.lower().endswith(("_n.png", "_s.png", "_e.png")):
                    continue

                src_png = root_path / file
                rel_path = src_png.relative_to(input_dir)
                block_key = str(rel_path.with_suffix("")).replace("\\", "/")
                block_name = src_png.stem

                try:
                    img = Image.open(src_png)
                except Exception as exc:
                    print(f"  [!] Skipping unreadable image {rel_path}: {exc}")
                    continue

                # 1. Resolve Material Profile (Cache -> Local Ollama VLM -> Heuristic Fallback)
                if block_key in profiles_cache:
                    profile = profiles_cache[block_key]
                else:
                    profile = None
                    if use_llm:
                        print(f"[*] Profiling '{block_name}' via Ollama ({model})...")
                        profile = query_ollama_vlm(img, block_name, model=model, ollama_url=ollama_url)
                    if profile is None:
                        profile = heuristic_profile(block_name)
                    profiles_cache[block_key] = profile
                    cache_dirty = True

                # 2. Generate LabPBR _n.png and _s.png
                n_img, s_img = build_labpbr_maps(img, profile, flip_y=flip_y)

                dest_folder = output_dir / rel_path.parent
                dest_folder.mkdir(parents=True, exist_ok=True)

                n_path = dest_folder / f"{block_name}_n.png"
                s_path = dest_folder / f"{block_name}_s.png"
                n_img.save(n_path, format="PNG")
                s_img.save(s_path, format="PNG")

                # 3. If the base texture is animated (.png.mcmeta), copy .mcmeta for _n and _s
                src_mcmeta = src_png.with_name(f"{file}.mcmeta")
                if src_mcmeta.exists():
                    shutil.copyfile(src_mcmeta, dest_folder / f"{block_name}_n.png.mcmeta")
                    shutil.copyfile(src_mcmeta, dest_folder / f"{block_name}_s.png.mcmeta")

                processed_count += 1
                if processed_count % 50 == 0:
                    print(f"  [+] Generated {processed_count} LabPBR pairs...")
                    if cache_dirty:
                        with open(cache_path, "w", encoding="utf-8") as f:
                            json.dump(profiles_cache, f, indent=2, sort_keys=True)
                        cache_dirty = False

    if cache_dirty:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(profiles_cache, f, indent=2, sort_keys=True)

    print(f"\n[✓] Done! Generated {processed_count} LabPBR (_n / _s) texture pairs in '{output_dir}'.")
    print(f"[✓] Material profiles saved to '{cache_path}'.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a 32x LabPBR 1.3 Add-on Pack.")
    parser.add_argument("-i", "--input", type=Path, required=True, help="Path to extracted base 32x resource pack")
    parser.add_argument("-o", "--output", type=Path, default=Path("build_addon"), help="Output directory for LabPBR add-on")
    parser.add_argument("-c", "--cache", type=Path, default=Path("material_profiles.json"), help="JSON profile cache path")
    parser.add_argument("--no-llm", action="store_true", help="Skip Ollama VLM and use fast deterministic heuristics only")
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
    )


if __name__ == "__main__":
    main()