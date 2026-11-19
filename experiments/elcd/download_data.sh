#!/usr/bin/env bash
# The data of the ELCD experiment, fetched or generated into data/ (git-ignored). Run from the repository root:
#
#   bash experiments/elcd/download_data.sh [DIR]          DIR defaults to data
#
# 1. DIR/lasa      The LASA handwriting dataset (Khansari-Zadeh and Billard, 2011): a git clone of its public
#                  repository at the commit the paper's runs read (about 21 MB). The dataset is free for
#                  non-commercial academic use and must not be modified or distributed without its authors'
#                  permission (DIR/lasa/Readme.txt), so it is not part of this repository.
# 2. DIR/pendulum  n-2.npz, n-4.npz, n-8.npz (3.3 MB): the pendulum data of Jaffe et al. (NeurIPS 2024), from their
#                  public repository (MIT licence) at a fixed commit.
# 3. DIR/npend     8-N5000.npz (9 MB): the eight-link pendulum demonstrations of the paper's runs. Nothing is
#                  downloaded: experiments/elcd/train/pendulum.py simulates them (NumPy only, about 4 CPU minutes).
#
# A step whose files exist is not repeated; the files are checked in every run (steps 1 and 2: SHA-256, an error
# if it differs; step 3: reported). PYTHON selects the interpreter of step 3 (default: python). The code reads
# data/ by default; for another DIR set ELCD_LASA_DIR=DIR/lasa/DataSet, ELCD_PENDULUM_DIR=DIR/pendulum and
# ELCD_NPEND_DIR=DIR/npend.
set -euo pipefail

DIR=${1:-data}
PYTHON=${PYTHON:-python}
LASA_URL=https://bitbucket.org/khansari/lasahandwritingdataset.git
LASA_COMMIT=38304f7c0ac4708b0fd38331d94d02095ad0ccfd
LASA_SHA256=cf70c381a70c634ac0a109c6dd7acdf17c6bc76f8cacbeb18c25f5da28a4841a    # the 30 .mat files, in C sort order
ELCD_URL=https://raw.githubusercontent.com/seanjaffe1/Extended-Linearized-Contracting-Dynamics
ELCD_COMMIT=87c171f428c93e199da3eb32a60e03bcaa940d93
NPEND_SHA256=2968ca39bfff2165af2694730af74f72b62e71aedc98d2ca716a8fa7c46039ef    # pos then vel of 8-N5000.npz, as bytes

[ -f experiments/elcd/train/pendulum.py ] || { echo "run this script from the repository root" >&2; exit 1; }

sha256() {      # the SHA-256 of standard input
    if command -v sha256sum > /dev/null; then sha256sum; else shasum -a 256; fi | cut -d ' ' -f 1
}

mkdir -p "$DIR"

# 1. LASA
if [ -d "$DIR/lasa/DataSet" ]; then
    echo "lasa: $DIR/lasa/DataSet exists, not downloaded"
else
    [ ! -e "$DIR/lasa" ] || { echo "lasa: $DIR/lasa exists without a DataSet folder: move it away and run again" >&2; exit 1; }
    rm -rf "$DIR/lasa.partial"
    git clone --quiet "$LASA_URL" "$DIR/lasa.partial"
    git -C "$DIR/lasa.partial" checkout --quiet --detach "$LASA_COMMIT"
    mv "$DIR/lasa.partial" "$DIR/lasa"
    echo "lasa: cloned $LASA_URL at $LASA_COMMIT into $DIR/lasa"
fi
got=$(find "$DIR/lasa/DataSet/" -name '*.mat' -print0 | LC_ALL=C sort -z | xargs -0 cat | sha256)
[ "$got" = "$LASA_SHA256" ] || { echo "lasa: the .mat files of $DIR/lasa/DataSet are not those of commit $LASA_COMMIT" >&2; exit 1; }
echo "lasa: 30 .mat files, SHA-256 as expected"

# 2. The pendulum data of Jaffe et al.
mkdir -p "$DIR/pendulum"
while read -r want name; do
    file="$DIR/pendulum/$name"
    if [ ! -f "$file" ]; then
        curl --fail --silent --show-error --location --retry 3 -o "$file.partial" \
            "$ELCD_URL/$ELCD_COMMIT/data/pendulum/$name"
        mv "$file.partial" "$file"
    fi
    [ "$(sha256 < "$file")" = "$want" ] || { echo "pendulum: $file does not have the expected SHA-256" >&2; exit 1; }
    echo "pendulum: $file, SHA-256 as expected"
done << 'EOF'
1855c7820c40169601f9d0ce98d2afdb08026f03f8775aaff00e447777fedef2 n-2.npz
bf8cd8fb92bb3d7f5eac691a1aa1ee7fee8f1d8aed7b15c570a0a5de83b26acc n-4.npz
837b5d2861561c65d05af391e92e8ffebc40d1284dded7d1720768814a843925 n-8.npz
EOF

# 3. The generated eight-link pendulum demonstrations (npend:8-N5000)
if [ -f "$DIR/npend/8-N5000.npz" ]; then
    echo "npend: $DIR/npend/8-N5000.npz exists, not generated"
else
    echo "npend: simulating 8-N5000 (about 4 CPU minutes)"
    rm -rf "$DIR/npend.partial"
    "$PYTHON" -c "import sys; from experiments.elcd.train.pendulum import generate_npend; generate_npend('8-N5000', cache_dir=sys.argv[1])" "$DIR/npend.partial"
    mkdir -p "$DIR/npend"
    mv "$DIR/npend.partial/8-N5000.npz" "$DIR/npend/8-N5000.npz"
    rmdir "$DIR/npend.partial"
fi
"$PYTHON" - "$DIR/npend/8-N5000.npz" "$NPEND_SHA256" << 'EOF'
import hashlib
import sys

import numpy as np

with np.load(sys.argv[1]) as z:
    got = hashlib.sha256(z["pos"].tobytes() + z["vel"].tobytes()).hexdigest()
    print(f"npend: {sys.argv[1]}, damping {float(z['damping']):.6f}, pos {z['pos'].shape}")
if got == sys.argv[2]:
    print("npend: pos and vel are bit-identical to the arrays the paper's runs read")
else:
    print("npend: NOTE: pos and vel are not bit-identical to the arrays the paper's runs read (damping there: "
          "4.891407). With the same damping the difference is rounding in the linear algebra library; runs on "
          "these arrays are then not comparable bit for bit with the paper's records.")
EOF
