# LabPBR 1.3 Resource Pack Generator

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
![LabPBR 1.3](https://img.shields.io/badge/Spec-LabPBR%201.3-00b4d8)
![Resolution](https://img.shields.io/badge/Resolution-Any%20(16x--512x%2B)-success)

An automated Python pipeline and interactive Web GUI that generates standalone **LabPBR 1.3** Physically Based Rendering (PBR) add-on resource packs (`*_n.png` normal/height/AO maps and `*_s.png` specular/smoothness/metal/SSS/emissive maps) for **any resolution** Minecraft Java Edition resource pack (`16x`, `32x`, `64x`, `128x`, `256x`, `512x+`).

Because the generator outputs a **standalone overlay add-on pack** containing only `_n.png`, `_s.png`, animation `.mcmeta`, and `pack.mcmeta` files, the original resource pack's diffuse textures are never altered, blurred, or redistributed. Simply stack the generated add-on pack directly above your base resource pack in Minecraft's Resource Pack menu.

---

## Overview

Converting a traditional Minecraft resource pack into a full LabPBR 1.3 pack manually requires authoring thousands of multi-channel normal, parallax height, cavity ambient occlusion, smoothness, metallic $F_0$, subsurface scattering, and emission maps.

**LabPBR 1.3 Resource Pack Generator** automates this process with **100% pixel-grid fidelity** using a two-stage hybrid architecture:

1. **Stage 1 — Material Semantic Profiling (Local Ollama Vision LLM + Deterministic Heuristics):**
   - Queries a local Vision Language Model via [Ollama](https://ollama.com/) (such as `qwen2.5vl:7b` or `llama3.2-vision:11b`) or uses a built-in deterministic heuristic rule engine (`--no-llm`) to classify each texture's physical material properties (`stone`, `wood`, `metal`, `gem`, `foliage`, `emissive`, `fabric`, `glass`, `liquid`, `sand`, `painting`, `banner`, `mob`) and composite masking rules.
   - Smart **material group deduplication** shares LLM profiles across block variants (`_stairs`, `_slab`, `_wall`), CTM/random tile sets (`0.png`, `1.png`, ...), and mob coat/biome variants, dramatically reducing LLM query time.
   - All material profiles are cached in `material_profiles.json` so subsequent runs are instantaneous.
2. **Stage 2 — Deterministic Pixel-Exact LabPBR Channel Synthesis (NumPy + Pillow):**
   - Synthesizes bit-exact RGBA normal (`_n.png`) and specular (`_s.png`) maps matching the exact width and height of every input sprite.
   - Separates composite secondary materials on a single sprite—such as glowing veins or mineral specks in ores, wooden handles/hafts on metal tools and weapons, gold trim on villager robes, or moss on iron golems—using chromaticity-outlier, luminance, saturation, and wood-hue masks.

---

## Features

- **Any-Resolution Support (`16x` to `512x+`):** Automatically detects the donor pack's native resolution and preserves 1:1 pixel alignment for every texture regardless of dimensions.
- **Full Asset Coverage:**
  - **Blocks & CTM/OptiFine Random Tiles:** Seamless toroidal (`wrap`) 3×3 Scharr normal filtering so adjacent world blocks tile without visible border seams.
  - **Animated Sprite Strips:** Automatically detects vertical animation strips (`height > width` multiples) and processes each `W × W` frame independently so adjacent animation frames never bleed into one another. Copies and sanitizes companion `.png.mcmeta` files.
  - **Items, Weapons, Tools & Armor:** Dedicated metallic $F_0$ assignments (`230` Iron, `231` Gold, `234` Copper, `255` Netherite/Tinted Metal) and gemstone reflections (`Diamond`, `Emerald`, `Amethyst`, `Ruby`) with automatic **wooden handle/pole masking** (`mask_mode: "wood_handle"`) so sword hilts, pickaxe handles, and spear shafts render as matte wood while blades gleam as polished metal. Includes 3D worn equipment (`entity/equipment/**` and `models/armor/**`).
  - **Paintings:** Automatically scans the pack's `painting/` directory to detect the pack-specific wooden frame border width in pixels. Renders the outer frame as beveled, relief-shaded polished wood (`smoothness: 138`, `F0: 6`) and the inner artwork canvas with a flat normal (`(128, 128, 255)`) and satin oil-paint sheen (`smoothness: 92`, `F0: 10`).
  - **Banners & Shields:** Separates wooden structural crossbars/stands from woven fabric and extracts high-pass micro-weave texture on high-resolution cloth while keeping painted banner emblems flat in the normal map.
  - **Mobs, Creatures, Animals & Villagers:** Covers all `assets/minecraft/textures/entity/**` and `assets/minecraft/optifine/random/entity/**` textures:
    - **UV-Island Margin Inpainting:** Eliminates false 3D box seams along transparent UV gutters on unfolded entity model sheets.
    - **Villagers, Professions & Badges:** Warm skin/cloth subsurface scattering (`SSS: 72`), metallic/leather profession attire accents (`armorer`, `weaponsmith`, `cleric`, `librarian`, `butcher`), and gem/metal profession level badges (`stone`, `iron`, `gold`, `emerald`, `diamond`).
    - **Golems, Animals, Aquatic & Nether Mobs:** Tailored profiles for matte fur/wool/feathers, wet mucous aquatic skin (`axolotl`, `frog`, `squid`, `fish`, `dolphin`), translucent gel (`slime`, `allay`, `vex`), chitin shells (`spider`, `shulker`, `armadillo`), and oxidized copper/iron golems.
    - **Emissive Eyes & Bioluminescent Overlays:** Automatically merges companion OptiFine `*_e.png` emissive masks (`iron_golem_e`, `spider_e`, `creaking_e`, `wolf_angry_e`) and dedicated glow overlays (`*_eyes.png`, `warden_bioluminescent_layer.png`, `warden_heart.png`) into the `_s.png` Alpha channel.
- **Minecraft Version Compatibility:** Automatically clones `pack.mcmeta` format fields (`pack_format`, `supported_formats`, or Minecraft 26.x+ `min_format`/`max_format`) and `pack.png` from the donor pack so Minecraft never warns that the add-on pack is incompatible or outdated.
- **Built-in Web GUI:** Zero-dependency local web interface (`web_ui.py` + `index.html`) with environment health checks, one-click dependency and Ollama model installation, native folder/file dialogs, real-time console streaming, and automatic `.zip` packaging.

---

## LabPBR 1.3 Channel Specification

Every generated texture pair strictly conforms to the [LabPBR 1.3 standard](https://shaderlabs.org/wiki/LabPBR_Material_Standard):

| Map | Channel | Range | Description |
| :--- | :--- | :--- | :--- |
| **`*_n.png`** | **Red (R)** | `0–255` | DirectX/OpenGL Tangent-Space Normal X (`128` = flat `0.0`) |
| **`*_n.png`** | **Green (G)** | `0–255` | Tangent-Space Normal Y (`128` = flat `0.0`; OpenGL `+Y` up default, `--flip-y` for `-Y`) |
| **`*_n.png`** | **Blue (B)** | `140–255` | Local 3×3 Cavity Ambient Occlusion (`255` = unoccluded) |
| **`*_n.png`** | **Alpha (A)** | `1–255` | Parallax Occlusion Mapping (POM) Height (`255` = flush surface; **never `0`**, which breaks POM in shaders) |
| **`*_s.png`** | **Red (R)** | `0–255` | Perceptual Smoothness (`0` = rough/matte, `255` = mirror smooth) |
| **`*_s.png`** | **Green (G)** | `0–229`<br>`230–237`<br>`255` | Dielectric $F_0$ reflectance (`0–229`), Hardcoded Predefined Metals (`230` Iron, `231` Gold, `232` Aluminum, `233` Chrome, `234` Copper, `235` Lead, `236` Platinum, `237` Silver), or Tinted Metal (`255`). *Never linearly blended across the `230` boundary.* |
| **`*_s.png`** | **Blue (B)** | `0–64`<br>`65–255` | Porosity / Hygroscopic WetnessDarkening (`0–64`) **or** Subsurface Scattering (`65–255`) |
| **`*_s.png`** | **Alpha (A)** | `0–254` | Emissive Light Strength (`0` = non-emissive, `1–254` = emission brightness; **never `255`**) |

---

## Prerequisites & Installation

1. **Python 3.9+** — [Download Python](https://www.python.org/downloads/)
2. **Required Python Packages** (`numpy`, `Pillow`, `requests`):
   ```bash
   pip install -r requirements.txt
   ```
   *(Note: You can also install missing Python packages with a single click inside the Web GUI.)*
3. **Optional — Ollama (for Vision LLM Material Classification):**
   - Download and install [Ollama](https://ollama.com/download) if you want AI-assisted material classification.
   - Pull a vision-capable model (recommended: `qwen2.5vl:7b`):
     ```bash
     ollama pull qwen2.5vl:7b
     ```
   - **No GPU / No Ollama?** You can run the generator anytime in deterministic heuristic mode (`--no-llm` or the **"Run without LLM"** checkbox in the Web GUI) with zero external services required.

---

## Running the Web GUI

The easiest way to configure and run the generator is through the built-in Web GUI.

### Launching on Windows
Double-click [`launch_gui.bat`](launch_gui.bat) in Explorer, or run from a terminal:
```bat
launch_gui.bat
```

### Launching Cross-Platform (Windows / macOS / Linux)
```bash
python web_ui.py
```
This starts a local server at `http://localhost:8765` and automatically opens your web browser.

### Web GUI Capabilities
1. **Environment Status Cards:**
   - Detects your Python version and provides a link to python.org if missing.
   - Checks for `numpy`, `Pillow`, and `requests`, with an **Install Missing Python Modules** button if any are absent.
   - Detects whether Ollama is installed and whether the local server (`http://localhost:11434`) is reachable.
   - Lists installed Ollama models (highlighting vision-capable models) and includes a one-click **Pull Selected Vision Model** downloader (`qwen2.5vl:7b`, `llama3.2-vision:11b`, `qwen2.5vl:3b`, `llava:7b`).
2. **Resource Pack Extraction & Directory Selection:**
   - Click **Select Resource Pack (.zip) to Extract** to pick any resource pack `.zip` file via a native file dialog and extract it directly into `./input_pack`.
   - Use the **Browse...** buttons to choose custom Input (`input_pack`) and Output (`build_addon`) directories.
   - Displays the auto-detected pack resolution badge (`16x`, `32x`, `64x`, etc.) and total diffuse texture count.
3. **Pipeline Flags & Auto-Packaging:**
   - Toggle **Run without LLM (`--no-llm`)**, **Refresh Profile Cache (`--refresh-cache`)**, or **Flip Normal Y Channel (`--flip-y`)**.
   - Enable **Automatically package output into a `.zip` resource pack** (defaults to `<input_pack_name>_pbr.zip`) so your finished add-on pack is ready to drop directly into Minecraft's `resourcepacks/` folder.
4. **Live Console Output:**
   - Streams real-time progress, Ollama verification diagnostics, and final summaries with a **Stop Process** button if you need to cancel a run.

### Web GUI Server CLI Flags (`web_ui.py`)
| Flag | Default | Description |
| :--- | :--- | :--- |
| `--port <PORT>` | `8765` | Local TCP port for the Web GUI HTTP server (`http://localhost:<PORT>`). |
| `--no-browser` | `False` | Start the Web GUI server without automatically opening a browser tab. |

---

## Command-Line Usage (`generate_labpbr.py`)

You can run the generator directly from the command line for scripting, batch processing, or headless workflows.

### Basic Examples

**1. Fast Deterministic Heuristic Mode (No Ollama / No LLM required):**
```bash
python generate_labpbr.py --input ./input_pack --output ./build_addon --no-llm
```

**2. Hybrid Vision LLM Mode (with Ollama `qwen2.5vl:7b`):**
```bash
python generate_labpbr.py --input ./input_pack --output ./build_addon --model qwen2.5vl:7b
```

**3. Force Re-evaluation of All Cached Material Profiles:**
```bash
python generate_labpbr.py --input ./input_pack --output ./build_addon --model qwen2.5vl:7b --refresh-cache
```

**4. Generate DirectX (`-Y`) Normal Maps Instead of OpenGL (`+Y`):**
```bash
python generate_labpbr.py --input ./input_pack --output ./build_addon --no-llm --flip-y
```

### Full CLI Flag Reference (`generate_labpbr.py`)

| Flag | Short | Default | Description |
| :--- | :--- | :--- | :--- |
| `--input <PATH>` | `-i` | *(Required)* | Path to the extracted donor Minecraft resource pack directory (must contain `assets/minecraft/textures/...`). |
| `--output <PATH>` | `-o` | `build_addon` | Output directory where the standalone LabPBR 1.3 add-on resource pack (`*_n.png`, `*_s.png`, `.mcmeta`, `pack.mcmeta`) will be written. |
| `--cache <PATH>` | `-c` | `material_profiles.json` | Path to the JSON file used to load and persist per-texture material classification profiles. |
| `--no-llm` | — | `False` | Skip querying Ollama and rely strictly on the fast deterministic heuristic classifier and existing cached profiles. |
| `--refresh-cache` | — | `False` | Ignore existing entries in `material_profiles.json` and recompute material profiles from scratch. |
| `--model <NAME>` | — | `qwen2.5vl:7b` | Name of the local Ollama vision model to query when LLM mode is active (e.g., `qwen2.5vl:7b`, `qwen2.5vl:3b`, `llama3.2-vision:11b`, `llava:7b`). |
| `--ollama-url <URL>` | — | `http://localhost:11434` | Base URL of the local or LAN Ollama HTTP server. |
| `--flip-y` | — | `False` | Invert the normal map Green (`Y`) channel for DirectX (`-Y` up) convention instead of the LabPBR 1.3 OpenGL (`+Y` up) default. |

---

## Using the Generated Pack in Minecraft

1. Copy either the generated `build_addon` folder or the auto-packaged `<pack_name>_pbr.zip` into your Minecraft `resourcepacks` directory.
2. In Minecraft, open **Options → Resource Packs** and move **both** your base resource pack and the LabPBR add-on pack to the **Selected** column, placing the **LabPBR Add-on Pack directly above** the base resource pack.
3. In your shader pack settings (e.g., *Complementary Reimagined/Unbound*, *BSL*, *Bliss*, *Kappa*, *Rethinking Voxels*, *SEUS PTGI*), set the **RP Support / Material Format** to **LabPBR 1.3** and enable Normal Maps, Specular Reflections, POM (Parallax Occlusion Mapping), and LabPBR Emissive pixels.

---

## License

This project is licensed under the **GNU General Public License v3.0 (GPLv3)** — see the [LICENSE](LICENSE) file for full details.
