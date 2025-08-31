#!/usr/bin/env python3
import argparse
import os
import shutil
import sys
import unicodedata
from pathlib import Path
from typing import Iterable, Optional, Tuple, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# Optional imports collected once
def _try_imports():
    mods = {}
    try:
        import PyPDF2
        mods["PyPDF2"] = PyPDF2
    except Exception:
        mods["PyPDF2"] = None
    try:
        import docx  # python-docx
        mods["docx"] = docx
    except Exception:
        mods["docx"] = None
    try:
        from PIL import Image, ImageOps, ImageFilter
        mods["PIL_Image"] = Image
        mods["PIL_ImageOps"] = ImageOps
        mods["PIL_ImageFilter"] = ImageFilter
        try:
            # guard huge images
            Image.MAX_IMAGE_PIXELS = 80_000_000
        except Exception:
            pass
    except Exception:
        mods["PIL_Image"] = None
        mods["PIL_ImageOps"] = None
        mods["PIL_ImageFilter"] = None
    try:
        import pytesseract
        mods["pytesseract"] = pytesseract
    except Exception:
        mods["pytesseract"] = None
    try:
        from pdfminer.high_level import extract_text as pdfminer_extract_text
        mods["pdfminer_extract_text"] = pdfminer_extract_text
    except Exception:
        mods["pdfminer_extract_text"] = None
    try:
        import pillow_heif  # enables HEIF open via PIL
        mods["pillow_heif"] = pillow_heif
        try:
            pillow_heif.register_heif_opener()
        except Exception:
            pass
    except Exception:
        mods["pillow_heif"] = None
    try:
        import filetype  # MIME sniff
        mods["filetype"] = filetype
    except Exception:
        mods["filetype"] = None
    return mods

MODS = _try_imports()

# ---------------- Keywords CNI FR ----------------
CNI_KEYWORDS = [
    "republique francaise",
    "république française",
    "carte nationale d'identite",
    "carte nationale d'identité",
    "carte nationale",
    "ministere de l'interieur",
    "ministère de l'intérieur",
    "ne le",
    "né le",
    "nee le",
    "né(e) le",
    "prenoms",
    "prénoms",
    "nom de naissance",
    "noms d'usage",
    "nationalite francaise",
    "nationalité française",
    "delivree le",
    "délivrée le",
    "valable jusqu'au",
    "sexe",
    "signature du titulaire",
    "autorite",
    "numero de document",
    "numéro de document",
    "date de naissance",
    "lieu de naissance",
]

def normalize_text(s: str) -> str:
    nfkd = unicodedata.normalize("NFKD", s)
    ascii_s = nfkd.encode("ASCII", "ignore").decode("ASCII")
    return ascii_s.lower()

NORM_KEYWORDS = {normalize_text(k) for k in CNI_KEYWORDS}

def text_has_cni_cues(text: str) -> bool:
    if not text:
        return False
    norm = normalize_text(text)
    return any(kw in norm for kw in NORM_KEYWORDS)

# ---------------- Extensions et types ----------------
TEXT_EXT = {".txt", ".md", ".csv", ".tsv", ".log", ".json", ".yaml", ".yml", ".ini", ".conf"}
PDF_EXT  = {".pdf"}
DOCX_EXT = {".docx"}
IMG_EXT = {
    ".jpg", ".jpeg", ".jfif", ".jpe",
    ".png",
    ".tif", ".tiff",
    ".gif",
    ".bmp",
    ".webp",
    ".pbm", ".pgm", ".ppm",
    ".heic", ".heif"
}
SCAN_EXT = TEXT_EXT | PDF_EXT | DOCX_EXT | IMG_EXT

# ---------------- Extracteurs ----------------
def read_text_plain(path: Path, encodings: Iterable[str] = ("utf-8", "latin-1", "cp1252")) -> str:
    for enc in encodings:
        try:
            return path.read_text(encoding=enc, errors="ignore")
        except Exception:
            continue
    return ""

def read_text_pdf(path: Path) -> str:
    text = ""
    if MODS.get("PyPDF2"):
        try:
            reader = MODS["PyPDF2"].PdfReader(str(path))
            for page in reader.pages:
                try:
                    text += page.extract_text() or ""
                except Exception:
                    continue
        except Exception:
            text = ""
    if not text and MODS.get("pdfminer_extract_text"):
        try:
            text = MODS["pdfminer_extract_text"](str(path)) or ""
        except Exception:
            pass
    return text

def read_text_docx(path: Path) -> str:
    if not MODS.get("docx"):
        return ""
    try:
        doc = MODS["docx"].Document(str(path))
        return "\n".join(p.text for p in doc.paragraphs)
    except Exception:
        return ""

# --- Image helpers ---
def _open_image_any(path: Path):
    if not MODS.get("PIL_Image"):
        return None
    Image = MODS["PIL_Image"]
    ImageOps = MODS["PIL_ImageOps"]
    ext = path.suffix.lower()
    try:
        im = Image.open(str(path))
    except Exception:
        # Try MIME sniffing if extension lies
        try:
            if MODS.get("filetype"):
                kind = MODS["filetype"].guess(str(path))
                if kind and kind.mime and kind.mime.startswith("image/"):
                    im = Image.open(str(path))
                else:
                    return None
            else:
                return None
        except Exception:
            return None

    # Fix EXIF orientation when possible
    try:
        if ImageOps is not None:
            im = ImageOps.exif_transpose(im)
    except Exception:
        pass
    return im

def _preprocess_for_ocr(im):
    # Grayscale, optional upscale if small, light binarization, sharpen
    ImageFilter = MODS["PIL_ImageFilter"]
    w, h = im.size
    scale = 1.0
    if max(w, h) < 1200:
        scale = 1200.0 / max(w, h)
    if scale > 1.0:
        im = im.resize((int(w * scale), int(h * scale)))
    im = im.convert("L")
    im = im.point(lambda x: 0 if x < 140 else x)
    if ImageFilter:
        im = im.filter(ImageFilter.SHARPEN)
    return im

def read_text_image(path: Path) -> str:
    if not (MODS.get("pytesseract") and MODS.get("PIL_Image")):
        return ""
    pytesseract = MODS["pytesseract"]

    im = _open_image_any(path)
    if im is None:
        return ""

    # For animated formats (GIF/WebP), we OCR the first frame only for speed
    try:
        im.seek(0)
    except Exception:
        pass

    pre = _preprocess_for_ocr(im)
    try:
        return pytesseract.image_to_string(pre, lang="fra+eng", config="--psm 6")
    except Exception:
        return ""

def extract_text_from_file(path: Path) -> str:
    suf = path.suffix.lower()
    if suf in TEXT_EXT:
        return read_text_plain(path)
    if suf in PDF_EXT:
        return read_text_pdf(path)
    if suf in DOCX_EXT:
        return read_text_docx(path)
    if suf in IMG_EXT:
        return read_text_image(path)
    return ""

def should_scan(path: Path) -> bool:
    suf = path.suffix.lower()
    if suf in SCAN_EXT:
        return True
    # If extension is odd, sniff MIME to catch image/pdf
    try:
        if MODS.get("filetype"):
            kind = MODS["filetype"].guess(str(path))
            if not kind:
                return False
            if kind.mime.startswith("image/"):
                return True
            if kind.mime == "application/pdf":
                return True
    except Exception:
        pass
    return False

# ---------------- Copies et MT ----------------
_copy_lock = Lock()
_print_lock = Lock()

def safe_copy(src: Path, dst_dir: Path) -> Optional[Path]:
    try:
        dst_dir.mkdir(parents=True, exist_ok=True)
        with _copy_lock:
            target = dst_dir / src.name
            if target.exists():
                stem, ext = target.stem, target.suffix
                i = 1
                while True:
                    candidate = dst_dir / f"{stem}({i}){ext}"
                    if not candidate.exists():
                        target = candidate
                        break
                    i += 1
        shutil.copy2(src, target)
        return target
    except Exception as e:
        with _print_lock:
            print(f"[WARN] Copy failed for {src}: {e}", file=sys.stderr)
        return None

def worker_process_file(p: Path, out_dir: Path, dry_run: bool) -> Tuple[Path, bool, Optional[Path]]:
    if not p.is_file() or not should_scan(p):
        return (p, False, None)
    try:
        txt = extract_text_from_file(p)
    except Exception as e:
        with _print_lock:
            print(f"[WARN] Extract failed for {p}: {e}", file=sys.stderr)
        return (p, False, None)
    if text_has_cni_cues(txt):
        copied_to = None
        if not dry_run:
            copied_to = safe_copy(p, out_dir)
        return (p, True, copied_to)
    return (p, False, None)

def scan_and_collect_parallel(src_root: Path, out_dir: Path, recurse: bool, dry_run: bool, workers: int) -> int:
    paths: List[Path] = []
    it = src_root.rglob("*") if recurse else src_root.glob("*")
    for p in it:
        if p.is_file() and should_scan(p):
            paths.append(p)

    hits = 0
    if not paths:
        return 0

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(worker_process_file, p, out_dir, dry_run) for p in paths]
        for fut in as_completed(futures):
            try:
                p, matched, copied_to = fut.result()
            except Exception as e:
                with _print_lock:
                    print(f"[WARN] Worker crashed: {e}", file=sys.stderr)
                continue
            if matched:
                hits += 1
                with _print_lock:
                    if dry_run:
                        print(f"[HIT] {p}")
                    else:
                        if copied_to:
                            print(f"[HIT] {p}\n[COPY] -> {copied_to}")
                        else:
                            print(f"[HIT] {p}\n[COPY] skipped or failed")
    return hits

# ---------------- Main CLI ----------------
def main():
    parser = argparse.ArgumentParser(
        description="Detection multithread de documents potentiellement CNI FR et copie dans fr_cni."
    )
    parser.add_argument("source", help="Dossier source a analyser")
    parser.add_argument("-o", "--out", default="fr_cni", help="Dossier de sortie (default: fr_cni)")
    parser.add_argument("-R", "--recursive", action="store_true", help="Scan recursif")
    parser.add_argument("--dry-run", action="store_true", help="Ne copie pas, affiche seulement les hits")
    parser.add_argument("--workers", type=int, default=max(4, (os.cpu_count() or 4) * 5),
                        help="Nb de threads travailleurs (default: 5x CPU cores)")
    args = parser.parse_args()

    src = Path(args.source).expanduser().resolve()
    if not src.exists() or not src.is_dir():
        print(f"[ERR] Dossier source invalide: {src}", file=sys.stderr)
        sys.exit(2)

    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Modules disponibles:")
    print(f"  PyPDF2: {'yes' if MODS.get('PyPDF2') else 'no'}")
    print(f"  pdfminer.six: {'yes' if MODS.get('pdfminer_extract_text') else 'no'}")
    print(f"  python-docx: {'yes' if MODS.get('docx') else 'no'}")
    print(f"  OCR (pytesseract+Pillow): {'yes' if MODS.get('pytesseract') and MODS.get('PIL_Image') else 'no'}")
    print(f"[INFO] Demarrage scan MT: {src} -> {out_dir} (recursive={args.recursive}, dry_run={args.dry_run}, workers={args.workers})")

    hits = scan_and_collect_parallel(src, out_dir, recurse=args.recursive, dry_run=args.dry_run, workers=args.workers)
    print(f"[DONE] Fichiers suspects: {hits}")
    sys.exit(0 if hits >= 0 else 1)

if __name__ == "__main__":
    main()

