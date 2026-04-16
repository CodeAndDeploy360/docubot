"""Parse uploaded files into raw text (and page hints for PDFs)."""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from typing import BinaryIO

import chardet
import pandas as pd
import pdfplumber
from pypdf import PdfReader


@dataclass
class ParsedDocument:
    """Raw text segments with optional page numbers (1-based for display)."""

    segments: list[tuple[str, int | None]]  # (text, page or None)


def _detect_encoding(data: bytes) -> str:
    guess = chardet.detect(data) or {}
    enc = guess.get("encoding")
    if enc:
        return enc
    try:
        data.decode("utf-8")
        return "utf-8"
    except UnicodeDecodeError:
        return "latin-1"


def _strip_bom(data: bytes) -> bytes:
    if data.startswith(b"\xef\xbb\xbf"):
        return data[3:]
    if data.startswith(b"\xff\xfe") or data.startswith(b"\xfe\xff"):
        return data[2:]
    return data


def parse_txt_bytes(data: bytes) -> ParsedDocument:
    data = _strip_bom(data)
    enc = _detect_encoding(data)
    text = data.decode(enc, errors="replace")
    return ParsedDocument([(text, None)])


def parse_csv_bytes(data: bytes, filename: str = "upload.csv") -> ParsedDocument:
    data = _strip_bom(data)
    enc = _detect_encoding(data)
    text_stream = io.StringIO(data.decode(enc, errors="replace"))
    # Handle ragged rows: use Python engine, on_bad_lines pad via manual read
    lines = text_stream.readlines()
    if not lines:
        return ParsedDocument([("", None)])
    buf = io.StringIO("".join(lines))
    reader = csv.reader(buf)
    rows: list[list[str]] = []
    max_len = 0
    for row in reader:
        max_len = max(max_len, len(row))
        rows.append(row)
    normalized: list[list[str]] = []
    for row in rows:
        if len(row) < max_len:
            row = row + [""] * (max_len - len(row))
        elif len(row) > max_len:
            row = row[:max_len]
        normalized.append(row)
    if not normalized:
        return ParsedDocument([("", None)])
    header = normalized[0]
    body = normalized[1:] if len(normalized) > 1 else []
    df = pd.DataFrame(body, columns=header[: len(normalized[0])])
    table = df.to_string(index=False)
    return ParsedDocument([(f"CSV file: {filename}\n\n{table}", None)])


def parse_pdf_bytes(data: bytes) -> ParsedDocument:
    segments: list[tuple[str, int | None]] = []
    # pdfplumber often preserves layout better for tables
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            t = page.extract_text() or ""
            if t.strip():
                segments.append((t, i))
    if segments:
        return ParsedDocument(segments)
    # Fallback: pypdf
    reader = PdfReader(io.BytesIO(data))
    for i, page in enumerate(reader.pages, start=1):
        t = page.extract_text() or ""
        if t.strip():
            segments.append((t, i))
    return ParsedDocument(segments if segments else [("", None)])


def parse_upload(filename: str, data: bytes) -> ParsedDocument:
    name = filename.lower()
    if name.endswith(".txt"):
        return parse_txt_bytes(data)
    if name.endswith(".csv"):
        return parse_csv_bytes(data, filename=filename)
    if name.endswith(".pdf"):
        return parse_pdf_bytes(data)
    raise ValueError(f"Unsupported file type: {filename}")


def parse_streamlit_uploaded_file(file) -> ParsedDocument:
    """Accept a Streamlit UploadedFile-like object with `.name` and `.getvalue()`."""
    return parse_upload(file.name, file.getvalue())
