#!/usr/bin/env python3
import argparse
import sys
import subprocess
import shutil
from pathlib import Path
import traceback

import pandas as pd

def log(msg):
    print(msg, flush=True)

def run(cmd, check=True):
    log(f"+ {' '.join(cmd)}")
    return subprocess.run(cmd, check=check)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def is_pdf(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() == ".pdf"

def find_pdfs(pdf_dir: Path):
    return [p for p in pdf_dir.rglob("*.pdf") if p.is_file()]

def download_from_drive(folder_id: str, dest: Path):
    ensure_dir(dest)
    # gdown can download entire folder by id
    cmd = ["gdown", "--folder", "--id", folder_id, "-O", str(dest), "--remaining-ok"]
    run(cmd)

def try_camelot(pdf_path: Path):
    try:
        import camelot
    except Exception as e:
        log(f"camelot not available: {e}")
        return []

    # try lattice then stream
    tables = []
    for flavor in ("lattice", "stream"):
        try:
            log(f"camelot reading ({flavor}) {pdf_path}")
            t = camelot.read_pdf(str(pdf_path), pages="all", flavor=flavor)
            if t and len(t) > 0:
                for tbl in t:
                    try:
                        df = tbl.df
                        # Clean header row heuristics: if first row looks like header, set header
                        if df.shape[0] > 1 and any(df.iloc[0].str.len() > 0):
                            df.columns = df.iloc[0]
                            df = df[1:].reset_index(drop=True)
                        tables.append(df)
                if tables:
                    break
        except Exception as e:
            log(f"camelot {flavor} failed on {pdf_path}: {e}")
    return tables

def try_pdfplumber(pdf_path: Path):
    try:
        import pdfplumber
    except Exception as e:
        log(f"pdfplumber not available: {e}")
        return []

    tables = []
    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            for page_idx, page in enumerate(pdf.pages, start=1):
                try:
                    raw_tables = page.extract_tables()
                    for raw in raw_tables:
                        # Normalize rows to same length
                        max_len = max((len(r) for r in raw), default=0)
                        norm = [(r + [None] * (max_len - len(r))) if r else [None] * max_len for r in raw]
                        df = pd.DataFrame(norm)
                        # Heuristic header
                        if df.shape[0] > 1 and any(pd.Series(df.iloc[0]).astype(str).str.len() > 0):
                            df.columns = df.iloc[0]
                            df = df[1:].reset_index(drop=True)
                        tables.append(df)
                except Exception as e:
                    log(f"pdfplumber failed on page {page_idx} of {pdf_path}: {e}")
    except Exception as e:
        log(f"pdfplumber open failed on {pdf_path}: {e}")
    return tables

def ocr_pdf(pdf_path: Path, ocr_lang: str, tmp_dir: Path) -> Path | None:
    ocrmypdf = shutil.which("ocrmypdf")
    if not ocrmypdf:
        log("ocrmypdf not installed; skipping OCR step.")
        return None
    out_pdf = tmp_dir / f"{pdf_path.stem}__ocr.pdf"
    try:
        run([ocrmypdf, "--skip-text", "--force-ocr", "-l", ocr_lang, str(pdf_path), str(out_pdf)])
        if out_pdf.exists():
            return out_pdf
    except subprocess.CalledProcessError as e:
        log(f"ocrmypdf failed: {e}")
    return None

def write_tables(dfs, base: str, out_dir: Path, per_table: bool, per_pdf: bool):
    written = []
    if per_table:
        for i, df in enumerate(dfs, start=1):
            out = out_dir / f"{base}__table-{i}.csv"
            df.to_csv(out, index=False)
            written.append(out)
    if per_pdf and dfs:
        # Concatenate with a separator column to preserve origin
        concat = []
        for i, df in enumerate(dfs, start=1):
            df2 = df.copy()
            df2.insert(0, "_table_index", i)
            concat.append(df2)
        merged = pd.concat(concat, axis=0, ignore_index=True)
        out = out_dir / f"{base}.csv"
        merged.to_csv(out, index=False)
        written.append(out)
    return written

def process_pdf(pdf_path: Path, out_dir: Path, per_table: bool, per_pdf: bool, allow_ocr: bool, ocr_lang: str, tmp_dir: Path):
    log(f"Processing: {pdf_path}")
    dfs = try_camelot(pdf_path)
    if not dfs:
        dfs = try_pdfplumber(pdf_path)
    if not dfs and allow_ocr:
        ocr_path = ocr_pdf(pdf_path, ocr_lang, tmp_dir)
        if ocr_path and ocr_path.exists():
            dfs = try_camelot(ocr_path) or try_pdfplumber(ocr_path)
    base = pdf_path.stem
    written = write_tables(dfs, base, out_dir, per_table, per_pdf)
    if dfs:
        log(f"Extracted {len(dfs)} tables from {pdf_path}. Wrote {len(written)} CSVs.")
    else:
        log(f"No tables found in {pdf_path}.")
    return written

def main():
    ap = argparse.ArgumentParser(description="Download PDFs from Google Drive and extract tables to CSV.")
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--drive-folder-id", type=str, help="Google Drive folder ID")
    group.add_argument("--drive-folder-url", type=str, help="Google Drive folder URL")
    ap.add_argument("--pdf-dir", type=str, default="data/pdfs", help="Directory to store/read PDFs")
    ap.add_argument("--output-dir", type=str, default="data/csvs", help="Directory to write CSVs")
    ap.add_argument("--per-table", action="store_true", help="Write one CSV per detected table")
    ap.add_argument("--per-pdf", action="store_true", help="Write one CSV per PDF (tables concatenated)")
    ap.add_argument("--ocr", action="store_true", help="Enable OCR fallback (uses ocrmypdf)")
    ap.add_argument("--ocr-lang", type=str, default="eng", help="Tesseract language code(s), e.g., 'eng', 'eng+spa'")
    ap.add_argument("--fail-on-empty", action="store_true", help="Exit non-zero if no tables extracted")
    args = ap.parse_args()

    pdf_dir = Path(args.pdf_dir)
    out_dir = Path(args.output_dir)
    tmp_dir = Path("tmp")
    ensure_dir(pdf_dir)
    ensure_dir(out_dir)
    ensure_dir(tmp_dir)

    if args.drive_folder_id or args.drive_folder_url:
        folder_id = args.drive_folder_id
        if not folder_id and args.drive_folder_url:
            # Accept both full URL and ID
            # Try to parse ID from URL
            try:
                parts = args.drive_folder_url.split("/folders/")[1]
                folder_id = parts.split("?")[0]
            except Exception:
                folder_id = None
        if folder_id:
            log(f"Downloading PDFs from Drive folder id={folder_id} into {pdf_dir}")
            try:
                download_from_drive(folder_id, pdf_dir)
            except Exception as e:
                log(f"Drive download failed: {e}")
                log(traceback.format_exc())

    pdfs = find_pdfs(pdf_dir)
    if not pdfs:
        log(f"No PDFs found under {pdf_dir}. Nothing to do.")
        sys.exit(1 if args.fail_on_empty else 0)

    total_written = 0
    for pdf_path in sorted(pdfs):
        try:
            written = process_pdf(
                pdf_path=pdf_path,
                out_dir=out_dir,
                per_table=args.per_table or (not args.per_pdf),
                per_pdf=args.per_pdf,
                allow_ocr=args.ocr,
                ocr_lang=args.ocr_lang,
                tmp_dir=tmp_dir,
            )
            total_written += len(written)
        except Exception as e:
            log(f"Error processing {pdf_path}: {e}")
            log(traceback.format_exc())

    if total_written == 0 and args.fail_on_empty:
        sys.exit(1)
    log(f"Done. Wrote {total_written} CSV file(s) to {out_dir}")

if __name__ == "__main__":
    main()
