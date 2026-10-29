#!/usr/bin/env python3
"""
Local Web GUI Backend for the LabPBR 1.3 Automated Resource Pack Generator.
Uses ONLY Python standard library modules so it can start even before third-party
dependencies (numpy, Pillow, requests) are installed, and can install them from the UI.
"""

import importlib.metadata
import importlib.util
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
import zipfile
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


WORKSPACE_DIR = Path(__file__).resolve().parent
INDEX_HTML_PATH = WORKSPACE_DIR / "index.html"
SOURCE_PACK_MARKER = ".source_pack_name"

REQUIRED_MODULES = [
    {"package": "numpy", "import_name": "numpy", "min_version": "1.26.0"},
    {"package": "Pillow", "import_name": "PIL", "min_version": "10.0.0"},
    {"package": "requests", "import_name": "requests", "min_version": "2.31.0"},
]

RECOMMENDED_VISION_MODELS = [
    {"name": "qwen2.5vl:7b", "label": "Qwen 2.5 VL (7B) — Recommended for Pixel Art & PBR"},
    {"name": "llama3.2-vision:11b", "label": "Llama 3.2 Vision (11B)"},
    {"name": "qwen2.5vl:3b", "label": "Qwen 2.5 VL (3B) — Lightweight / Fast VRAM"},
    {"name": "llava:7b", "label": "LLaVA (7B)"},
]

SCAN_SUBDIRS = (
    "assets/minecraft/textures/block",
    "assets/minecraft/textures/blocks",
    "assets/minecraft/textures/item",
    "assets/minecraft/textures/items",
    "assets/minecraft/textures/painting",
    "assets/minecraft/textures/paintings",
    "assets/minecraft/textures/entity",
    "assets/minecraft/optifine/random/entity",
    "assets/minecraft/textures/models/armor",
)


class JobManager:
    """Manages background CLI tasks (generator execution, pip install, ollama pull, zip extraction) and console logs."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.logs: List[str] = [
            "[*] LabPBR 1.3 Generator Web UI initialized.",
            f"[*] Workspace: {WORKSPACE_DIR}",
        ]
        self.status: str = "idle"  # idle | running | completed | error
        self.task_name: str = ""
        self.command_str: str = ""
        self.process: Optional[subprocess.Popen] = None
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None

    def log(self, message: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        with self.lock:
            for line in message.splitlines():
                self.logs.append(f"[{timestamp}] {line}")
            if len(self.logs) > 5000:
                self.logs = self.logs[-5000:]

    def clear_logs(self) -> None:
        with self.lock:
            self.logs.clear()

    def is_running(self) -> bool:
        with self.lock:
            return self.status == "running"

    def stop_current_job(self) -> bool:
        with self.lock:
            proc = self.process
            if proc is not None and proc.poll() is None:
                try:
                    proc.terminate()
                except Exception:
                    pass
                self.logs.append(f"[{time.strftime('%H:%M:%S')}] [!] Stop requested by user; terminating process...")
                return True
            return False

    def get_snapshot(self, since: int = 0) -> Dict[str, Any]:
        with self.lock:
            new_logs = self.logs[since:] if since < len(self.logs) else []
            elapsed = 0.0
            if self.start_time is not None:
                end_ref = self.end_time if self.end_time is not None else time.time()
                elapsed = max(0.0, end_ref - self.start_time)
            return {
                "status": self.status,
                "task_name": self.task_name,
                "command": self.command_str,
                "next_index": len(self.logs),
                "logs": new_logs,
                "elapsed_seconds": round(elapsed, 1),
            }

    def run_subprocess_job(
        self,
        task_name: str,
        cmd: List[str],
        post_hook: Optional[Any] = None,
    ) -> bool:
        with self.lock:
            if self.status == "running":
                return False
            self.status = "running"
            self.task_name = task_name
            self.command_str = " ".join(f'"{c}"' if " " in c else c for c in cmd)
            self.start_time = time.time()
            self.end_time = None

        self.log(f"=== Starting Task: {task_name} ===")
        self.log(f"$ {self.command_str}")

        def _worker() -> None:
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"
            env["PYTHONUTF8"] = "1"
            try:
                proc = subprocess.Popen(
                    cmd,
                    cwd=str(WORKSPACE_DIR),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                    env=env,
                )
                with self.lock:
                    self.process = proc

                assert proc.stdout is not None
                for raw_line in proc.stdout:
                    cleaned = raw_line.rstrip("\r\n")
                    if cleaned:
                        cleaned = re.sub(r"\x1b\[[0-9;?]*[a-zA-Z]", "", cleaned)
                        if cleaned.strip():
                            self.log(cleaned)

                rc = proc.wait()
                if rc == 0 and post_hook is not None:
                    post_hook()

                with self.lock:
                    self.process = None
                    self.end_time = time.time()
                    if rc == 0:
                        self.status = "completed"
                    else:
                        self.status = "error"

                if rc == 0:
                    self.log(f"[✓] Task '{task_name}' finished successfully (exit code 0).")
                else:
                    self.log(f"[!] Task '{task_name}' exited with code {rc}.")
            except Exception as exc:
                with self.lock:
                    self.process = None
                    self.end_time = time.time()
                    self.status = "error"
                self.log(f"[!] Exception during '{task_name}': {exc}")

        threading.Thread(target=_worker, daemon=True).start()
        return True


JOB_MANAGER = JobManager()


# =====================================================================
# Environment & Pack Inspection Helpers (Pure Stdlib)
# =====================================================================
def check_python_modules() -> Dict[str, Any]:
    results = []
    all_present = True
    for mod in REQUIRED_MODULES:
        spec = importlib.util.find_spec(mod["import_name"])
        installed = spec is not None
        version_str = None
        if installed:
            try:
                version_str = importlib.metadata.version(mod["package"])
            except Exception:
                version_str = "installed"
        else:
            all_present = False
        results.append({
            "package": mod["package"],
            "import_name": mod["import_name"],
            "min_version": mod["min_version"],
            "installed": installed,
            "version": version_str,
        })
    return {
        "all_installed": all_present,
        "modules": results,
    }


def find_ollama_executable() -> Optional[str]:
    which_path = shutil.which("ollama")
    if which_path:
        return which_path

    candidates = [
        Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe",
        Path(os.environ.get("PROGRAMFILES", "")) / "Ollama" / "ollama.exe",
        Path.home() / "AppData" / "Local" / "Programs" / "Ollama" / "ollama.exe",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return None


def check_ollama_status(ollama_url: str) -> Dict[str, Any]:
    exe_path = find_ollama_executable()
    installed = exe_path is not None
    base_url = ollama_url.rstrip("/")

    running = False
    models: List[Dict[str, Any]] = []
    error_msg: Optional[str] = None

    try:
        req = urllib.request.Request(f"{base_url}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            running = True
            installed = True
            for m in data.get("models", []):
                if not isinstance(m, dict):
                    continue
                name = str(m.get("name", ""))
                details = m.get("details", {}) if isinstance(m.get("details"), dict) else {}
                families = details.get("families") or []
                is_vision = any(
                    k in name.lower() or any(k in str(f).lower() for f in families)
                    for k in ("vl", "vision", "llava", "clip", "minicpm", "moondream", "gemma3", "pixtral")
                )
                size_gb = round(float(m.get("size", 0)) / (1024 ** 3), 2)
                models.append({
                    "name": name,
                    "size_gb": size_gb,
                    "parameter_size": details.get("parameter_size", ""),
                    "quantization": details.get("quantization_level", ""),
                    "is_vision": is_vision,
                })
    except Exception as exc:
        error_msg = str(exc)

    return {
        "installed": installed,
        "executable": exe_path,
        "download_url": "https://ollama.com/download",
        "running": running,
        "ollama_url": base_url,
        "models": models,
        "has_models": len(models) > 0,
        "has_vision_model": any(m["is_vision"] for m in models),
        "recommended_models": RECOMMENDED_VISION_MODELS,
        "error": error_msg,
    }


def _read_png_dimensions(png_path: Path) -> Optional[Tuple[int, int]]:
    """Reads PNG width and height directly from the 24-byte IHDR header using pure stdlib."""
    try:
        with open(png_path, "rb") as f:
            header = f.read(24)
        if len(header) >= 24 and header[:8] == b"\x89PNG\r\n\x1a\n" and header[12:16] == b"IHDR":
            w, h = struct.unpack(">II", header[16:24])
            return int(w), int(h)
    except Exception:
        pass
    return None


def derive_pack_base_name(input_dir_str: str) -> str:
    input_path = Path(input_dir_str)
    if not input_path.is_absolute():
        input_path = (WORKSPACE_DIR / input_path).resolve()

    marker_file = input_path / SOURCE_PACK_MARKER
    if marker_file.is_file():
        try:
            saved_name = marker_file.read_text(encoding="utf-8").strip()
            if saved_name:
                return saved_name
        except Exception:
            pass

    mcmeta_file = input_path / "pack.mcmeta"
    if mcmeta_file.is_file():
        try:
            data = json.loads(mcmeta_file.read_text(encoding="utf-8"))
            if "dokucraft" in data and isinstance(data["dokucraft"], dict):
                ver = data["dokucraft"].get("game_version", "")
                return f"dokucraft_{ver}".strip("_") if ver else "dokucraft"
        except Exception:
            pass

    folder_name = input_path.name
    return folder_name if folder_name and folder_name != "." else "resource_pack"


def inspect_input_pack(input_dir_str: str) -> Dict[str, Any]:
    input_path = Path(input_dir_str)
    if not input_path.is_absolute():
        input_path = (WORKSPACE_DIR / input_path).resolve()

    exists = input_path.is_dir()
    has_mcmeta = (input_path / "pack.mcmeta").is_file()
    total_textures = 0
    sampled_widths: List[int] = []

    if exists:
        for sub in SCAN_SUBDIRS:
            folder = input_path / sub
            if folder.is_dir():
                for root, _, files in os.walk(folder):
                    root_p = Path(root)
                    for f in files:
                        fl = f.lower()
                        if fl.endswith(".png") and not fl.endswith(("_n.png", "_s.png", "_e.png")):
                            total_textures += 1
                            if len(sampled_widths) < 60 and ("block" in sub or "item" in sub):
                                dims = _read_png_dimensions(root_p / f)
                                if dims and dims[0] > 0 and (dims[0] == dims[1] or dims[1] % dims[0] == 0):
                                    sampled_widths.append(dims[0])

    detected_res = Counter(sampled_widths).most_common(1)[0][0] if sampled_widths else None
    base_name = derive_pack_base_name(input_dir_str)
    default_zip = f"{base_name}_pbr.zip"

    return {
        "path": str(input_path),
        "exists": exists,
        "has_pack_mcmeta": has_mcmeta,
        "block_texture_count": total_textures,
        "resolution": detected_res,
        "base_name": base_name,
        "default_zip_name": default_zip,
    }


def extract_resource_pack_zip(zip_path: Path, target_input_dir: Path) -> Tuple[bool, str, str]:
    """
    Extracts a .zip resource pack into target_input_dir.
    If the zip wraps all contents inside a single top-level folder, unwraps it automatically
    so pack.mcmeta and assets/ sit directly inside target_input_dir.
    """
    if not zip_path.is_file():
        return False, f"Zip file not found: {zip_path}", ""

    pack_stem = zip_path.stem
    JOB_MANAGER.log(f"[*] Extracting resource pack '{zip_path.name}' into '{target_input_dir}'...")

    try:
        if target_input_dir.exists():
            shutil.rmtree(target_input_dir)
        target_input_dir.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(target_input_dir)

        # Check if pack.mcmeta is inside a single nested top-level folder
        if not (target_input_dir / "pack.mcmeta").exists():
            entries = [e for e in target_input_dir.iterdir() if e.name != SOURCE_PACK_MARKER]
            if len(entries) == 1 and entries[0].is_dir() and (entries[0] / "pack.mcmeta").exists():
                nested_root = entries[0]
                JOB_MANAGER.log(f"[*] Unwrapping nested root directory '{nested_root.name}/'...")
                for child in list(nested_root.iterdir()):
                    shutil.move(str(child), str(target_input_dir / child.name))
                nested_root.rmdir()

        (target_input_dir / SOURCE_PACK_MARKER).write_text(pack_stem, encoding="utf-8")
        info = inspect_input_pack(str(target_input_dir))
        res_tag = f"{info['resolution']}x, " if info.get("resolution") else ""
        msg = (
            f"Extracted '{zip_path.name}' -> '{target_input_dir}' "
            f"({res_tag}{info['block_texture_count']} textures found)."
        )
        JOB_MANAGER.log(f"[✓] {msg}")
        return True, msg, f"{pack_stem}_pbr.zip"
    except Exception as exc:
        err = f"Failed to extract zip: {exc}"
        JOB_MANAGER.log(f"[!] {err}")
        return False, err, ""


def create_output_addon_zip(output_dir: Path, zip_filename: str) -> Path:
    """
    Packages the generated LabPBR add-on directory (pack.mcmeta, pack.png, assets/)
    into a clean .zip ready to drop into Minecraft's resourcepacks/ folder.
    """
    if not zip_filename.lower().endswith(".zip"):
        zip_filename += ".zip"

    zip_dest = Path(zip_filename)
    if not zip_dest.is_absolute():
        zip_dest = (WORKSPACE_DIR / zip_dest).resolve()

    zip_dest.parent.mkdir(parents=True, exist_ok=True)
    JOB_MANAGER.log(f"[*] Packaging LabPBR add-on into '{zip_dest}'...")

    file_count = 0
    with zipfile.ZipFile(zip_dest, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(output_dir):
            root_path = Path(root)
            for file in sorted(files):
                full_path = root_path / file
                if full_path.suffix.lower() == ".zip" or full_path.resolve() == zip_dest.resolve():
                    continue
                rel_arc = full_path.relative_to(output_dir).as_posix()
                zf.write(full_path, arcname=rel_arc)
                file_count += 1

    size_mb = zip_dest.stat().st_size / (1024 * 1024)
    JOB_MANAGER.log(f"[✓] Created '{zip_dest.name}' ({file_count} files, {size_mb:.2f} MB) at: {zip_dest}")
    return zip_dest


def open_native_dialog(dialog_type: str, initial_dir: str = "") -> Optional[str]:
    """Opens a native Windows folder or file picker dialog via tkinter."""
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)

        start_dir = str(WORKSPACE_DIR)
        if initial_dir:
            candidate = Path(initial_dir)
            if not candidate.is_absolute():
                candidate = (WORKSPACE_DIR / candidate).resolve()
            if candidate.is_dir():
                start_dir = str(candidate)
            elif candidate.parent.is_dir():
                start_dir = str(candidate.parent)

        selected: Optional[str] = None
        if dialog_type == "directory":
            selected = filedialog.askdirectory(initialdir=start_dir, title="Select Folder")
        elif dialog_type == "zip_file":
            selected = filedialog.askopenfilename(
                initialdir=start_dir,
                title="Select Resource Pack (.zip)",
                filetypes=[("Minecraft Resource Pack (.zip)", "*.zip"), ("All Files", "*.*")],
            )
        elif dialog_type == "json_file":
            selected = filedialog.asksaveasfilename(
                initialdir=start_dir,
                title="Select Material Profile Cache (.json)",
                defaultextension=".json",
                filetypes=[("JSON Cache File", "*.json"), ("All Files", "*.*")],
            )
        root.destroy()
        if selected:
            sel_path = Path(selected).resolve()
            try:
                rel = sel_path.relative_to(WORKSPACE_DIR)
                return f"./{rel.as_posix()}"
            except ValueError:
                return str(sel_path)
        return None
    except Exception as exc:
        JOB_MANAGER.log(f"[!] Native file dialog error: {exc}")
        return None


# =====================================================================
# HTTP Request Handler
# =====================================================================
class LabPBRWebHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        pass

    def _send_json(self, payload: Dict[str, Any], status_code: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def _read_json_body(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        return json.loads(raw.decode("utf-8"))

    def do_OPTIONS(self) -> None:
        self._send_json({"ok": True})

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        route = parsed.path
        query = urllib.parse.parse_qs(parsed.query)

        if route in ("/", "/index.html"):
            if INDEX_HTML_PATH.is_file():
                content = INDEX_HTML_PATH.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
            else:
                self.send_error(404, "index.html not found")
            return

        if route == "/api/status":
            ollama_url = query.get("ollama_url", ["http://localhost:11434"])[0]
            input_dir = query.get("input_dir", ["./input_pack"])[0]
            status_payload = {
                "python": {
                    "installed": True,
                    "version": sys.version.split()[0],
                    "executable": sys.executable,
                    "download_url": "https://www.python.org/downloads/",
                },
                "modules": check_python_modules(),
                "ollama": check_ollama_status(ollama_url),
                "input_pack": inspect_input_pack(input_dir),
                "job": JOB_MANAGER.get_snapshot(since=0),
            }
            self._send_json(status_payload)
            return

        if route == "/api/logs":
            since = int(query.get("since", ["0"])[0])
            self._send_json(JOB_MANAGER.get_snapshot(since=since))
            return

        self.send_error(404, "Endpoint not found")

    def do_POST(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        route = parsed.path
        body = self._read_json_body()

        if route == "/api/browse":
            dialog_type = str(body.get("type", "directory"))
            initial_dir = str(body.get("initial_dir", ""))
            selected = open_native_dialog(dialog_type=dialog_type, initial_dir=initial_dir)
            extra: Dict[str, Any] = {}
            if selected and dialog_type == "directory":
                extra["input_pack"] = inspect_input_pack(selected)
            self._send_json({"selected": selected, **extra})
            return

        if route == "/api/extract-pack":
            if JOB_MANAGER.is_running():
                self._send_json({"ok": False, "error": "Another task is currently running."}, 409)
                return

            zip_path_str = str(body.get("zip_path", "")).strip()
            input_dir_str = str(body.get("input_dir", "./input_pack")).strip() or "./input_pack"

            if not zip_path_str:
                picked = open_native_dialog("zip_file")
                if not picked:
                    self._send_json({"ok": False, "cancelled": True})
                    return
                zip_path_str = picked

            zip_path = Path(zip_path_str)
            if not zip_path.is_absolute():
                zip_path = (WORKSPACE_DIR / zip_path).resolve()

            target_dir = Path(input_dir_str)
            if not target_dir.is_absolute():
                target_dir = (WORKSPACE_DIR / target_dir).resolve()

            ok, message, default_zip = extract_resource_pack_zip(zip_path, target_dir)
            self._send_json({
                "ok": ok,
                "message": message,
                "zip_path": str(zip_path),
                "default_zip_name": default_zip,
                "input_pack": inspect_input_pack(str(target_dir)),
            })
            return

        if route == "/api/install-modules":
            cmd = [sys.executable, "-m", "pip", "install", "numpy>=1.26.0", "Pillow>=10.0.0", "requests>=2.31.0"]
            started = JOB_MANAGER.run_subprocess_job("Install Required Python Modules", cmd)
            self._send_json({"ok": started})
            return

        if route == "/api/start-ollama":
            exe = find_ollama_executable()
            if not exe:
                self._send_json({"ok": False, "error": "Ollama executable not found."}, 404)
                return
            try:
                subprocess.Popen(
                    [exe, "serve"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                JOB_MANAGER.log("[*] Started 'ollama serve' background daemon.")
                time.sleep(1.5)
                self._send_json({"ok": True})
            except Exception as exc:
                self._send_json({"ok": False, "error": str(exc)}, 500)
            return

        if route == "/api/pull-model":
            model_name = str(body.get("model", "qwen2.5vl:7b")).strip()
            exe = find_ollama_executable() or "ollama"
            cmd = [exe, "pull", model_name]
            started = JOB_MANAGER.run_subprocess_job(f"Download Ollama Model ({model_name})", cmd)
            self._send_json({"ok": started})
            return

        if route == "/api/execute":
            if JOB_MANAGER.is_running():
                self._send_json({"ok": False, "error": "A process is already running."}, 409)
                return

            input_dir = str(body.get("input_dir", "./input_pack")).strip() or "./input_pack"
            output_dir = str(body.get("output_dir", "./build_addon")).strip() or "./build_addon"
            cache_file = str(body.get("cache_file", "material_profiles.json")).strip() or "material_profiles.json"
            no_llm = bool(body.get("no_llm", False))
            refresh_cache = bool(body.get("refresh_cache", False))
            flip_y = bool(body.get("flip_y", False))
            model_name = str(body.get("model", "qwen2.5vl:7b")).strip() or "qwen2.5vl:7b"
            ollama_url = str(body.get("ollama_url", "http://localhost:11434")).strip() or "http://localhost:11434"
            auto_zip = bool(body.get("auto_zip", False))
            zip_name = str(body.get("zip_name", "")).strip()

            cmd = [
                sys.executable,
                "-u",
                str(WORKSPACE_DIR / "generate_labpbr.py"),
                "--input",
                input_dir,
                "--output",
                output_dir,
                "--cache",
                cache_file,
                "--model",
                model_name,
                "--ollama-url",
                ollama_url,
            ]
            if no_llm:
                cmd.append("--no-llm")
            if refresh_cache:
                cmd.append("--refresh-cache")
            if flip_y:
                cmd.append("--flip-y")

            out_path = Path(output_dir)
            if not out_path.is_absolute():
                out_path = (WORKSPACE_DIR / out_path).resolve()

            if not zip_name:
                zip_name = f"{derive_pack_base_name(input_dir)}_pbr.zip"

            def _post_hook() -> None:
                if auto_zip:
                    create_output_addon_zip(out_path, zip_name)

            started = JOB_MANAGER.run_subprocess_job("Generate LabPBR 1.3 Resource Pack", cmd, post_hook=_post_hook)
            self._send_json({"ok": started})
            return

        if route == "/api/stop":
            stopped = JOB_MANAGER.stop_current_job()
            self._send_json({"ok": stopped})
            return

        if route == "/api/clear-logs":
            JOB_MANAGER.clear_logs()
            self._send_json({"ok": True})
            return

        self.send_error(404, "Endpoint not found")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Launch the LabPBR 1.3 Generator Web GUI.")
    parser.add_argument("--port", type=int, default=8765, help="Local port for the Web GUI server (default: 8765)")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically open the browser on startup")
    args = parser.parse_args()

    server = ThreadingHTTPServer(("127.0.0.1", args.port), LabPBRWebHandler)
    url = f"http://localhost:{args.port}"
    print(f"[*] LabPBR 1.3 Web GUI Server running at: {url}")
    print("[*] Press Ctrl+C to stop.")

    if not args.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[*] Shutting down Web GUI server.")
        server.server_close()


if __name__ == "__main__":
    main()
