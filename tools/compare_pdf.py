"""Compare regenerated figure PDFs with the committed ones.

    python tools/compare_pdf.py                 # every tracked PDF whose bytes differ from HEAD
    python tools/compare_pdf.py NEW.pdf REF.pdf # two files

A regenerated PDF never has the bytes of the committed one (the creation date is stored in it), so the comparison
is on the extracted text (pdftotext -layout) and on the pages rendered at 150 dpi (pdftoppm): the number of pixels
that differ and the largest difference of a channel (0 to 255). Needs git, pdftotext and pdftoppm (poppler-utils),
NumPy and Pillow. Exit status 1 when a text differs or more than --max-fraction of the pixels differ.
"""
import argparse
import glob
import os
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image


def text_of(pdf):
    return subprocess.run(["pdftotext", "-layout", pdf, "-"], check=True, capture_output=True, text=True).stdout


def pages_of(pdf, folder, tag):
    subprocess.run(["pdftoppm", "-r", "150", "-png", pdf, os.path.join(folder, tag)], check=True)
    return [np.asarray(Image.open(p).convert("RGB"), dtype=np.int16) for p in sorted(glob.glob(os.path.join(folder, tag + "*.png")))]


def compare(new, ref, tmp):
    same_text = text_of(new) == text_of(ref)
    a, b = pages_of(new, tmp, "new"), pages_of(ref, tmp, "ref")
    for p in glob.glob(os.path.join(tmp, "*.png")):
        os.remove(p)
    if len(a) != len(b) or any(x.shape != y.shape for x, y in zip(a, b)):
        return same_text, 1.0, 255, f"{[x.shape[:2] for x in a]} against {[y.shape[:2] for y in b]}"
    diff = [np.abs(x - y).max(axis=2) for x, y in zip(a, b)]
    n = sum(d.size for d in diff)
    return same_text, sum(int((d > 0).sum()) for d in diff) / n, max(int(d.max()) for d in diff), f"{len(a)} page(s), {n} px"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("pdf", nargs="*", help="NEW.pdf REF.pdf; nothing: the tracked PDFs that differ from HEAD")
    ap.add_argument("--max-fraction", type=float, default=0.0, help="largest accepted fraction of differing pixels")
    a = ap.parse_args()
    bad = 0
    with tempfile.TemporaryDirectory() as tmp:
        if a.pdf:
            if len(a.pdf) != 2:
                ap.error("give two files, or none")
            pairs = [(a.pdf[0], a.pdf[1], a.pdf[0])]
        else:
            changed = subprocess.run(["git", "diff", "--name-only", "HEAD", "--", "*.pdf"], check=True, capture_output=True,
                                     text=True).stdout.split()
            pairs = []
            for i, path in enumerate(changed):
                ref = os.path.join(tmp, f"ref{i}.pdf")
                with open(ref, "wb") as f:
                    f.write(subprocess.run(["git", "show", f"HEAD:{path}"], check=True, capture_output=True).stdout)
                pairs.append((path, ref, path))
            if not pairs:
                print("no tracked PDF differs from HEAD")
        for new, ref, name in pairs:
            same_text, frac, worst, note = compare(new, ref, tmp)
            ok = same_text and frac <= a.max_fraction
            bad += not ok
            print(f"{'ok  ' if ok else 'DIFF'} {name}: text {'identical' if same_text else 'DIFFERS'}, "
                  f"{100 * frac:.4f} % of the pixels differ (largest difference {worst}); {note}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
