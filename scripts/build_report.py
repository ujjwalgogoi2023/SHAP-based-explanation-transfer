"""Build the report as .docx and convert it to .pdf.

    python scripts/build_report.py

The PDF is produced by driving the installed Word through COM, which is the
only converter on this machine that reproduces the document's tables and
embedded figures faithfully. If Word is not available the .docx is still
written and the conversion is reported as skipped rather than failing the
build.
"""
from __future__ import annotations

import os
import subprocess
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.eda.report import DOCX, build  # noqa: E402
from src.eda.settings import ROOT  # noqa: E402

PDF = DOCX.replace(".docx", ".pdf")

# The document is opened read-write so Word can resolve its fields, repaginate
# and save the page numbers back into the .docx before the PDF is exported, so
# both deliverables carry the same numbering.
PS_CONVERT = r"""
$ErrorActionPreference = 'Stop'
$word = $null
try {{
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    $word.DisplayAlerts = 0
    $doc = $word.Documents.Open('{docx}')
    $doc.Fields.Update() | Out-Null
    $doc.Repaginate()
    $doc.Save()
    $doc.SaveAs([ref]'{pdf}', [ref]17)
    $doc.Close($false)
    Write-Output 'PDF_OK'
}} catch {{
    Write-Output ('PDF_FAIL: ' + $_.Exception.Message)
}} finally {{
    if ($word -ne $null) {{ $word.Quit() }}
}}
"""


def to_pdf(docx: str, pdf: str) -> bool:
    script = PS_CONVERT.format(docx=docx.replace("/", "\\"), pdf=pdf.replace("/", "\\"))
    out = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
                         capture_output=True, text=True)
    ok = "PDF_OK" in out.stdout
    if not ok:
        print("  PDF conversion skipped:", (out.stdout + out.stderr).strip()[:300])
    return ok


def main() -> None:
    docx = build()
    if to_pdf(docx, PDF):
        size = os.path.getsize(PDF) / 1e6
        print(f"  report -> {os.path.relpath(PDF, ROOT)}  ({size:.1f} MB)")


if __name__ == "__main__":
    main()
