#!/usr/bin/env bash
# The EigenWorms data of the contracting-SSM experiment, fetched into data/eigenworms (git-ignored). Run from the
# repository root:
#
#   bash experiments/appendix/contracting_ssm/download_data.sh [DIR]          DIR defaults to data/eigenworms
#
# Three pickles (107 MB) of the "neuralrde_split" (the train/validation/test split of Morrill et al., 2021) as the
# DEER repository (Lim et al., BSD 3-Clause licence) ships them, at a fixed commit. Each file is a list [x, y] of
# torch tensors, x of shape (N, 17984, 6) float32 and y of shape (N,) int64, with N = 181, 39, 39; reading them needs
# torch. The paper's runs read these files; the SHA-256 below are those of the files they read. The code reads
# data/eigenworms by default; for another DIR set EIGENWORMS_DIR=DIR.
set -euo pipefail

DIR=${1:-data/eigenworms}
URL=https://raw.githubusercontent.com/machine-discovery/deer
COMMIT=409140f
SRC=experiments/04_rnn_eigenworms/neuralrde_split

[ -f experiments/appendix/contracting_ssm/eigenworms_data.py ] || {
    echo "run this script from the repository root" >&2; exit 1; }

sha256() {      # the SHA-256 of a file
    if command -v sha256sum > /dev/null; then sha256sum "$1"; else shasum -a 256 "$1"; fi | cut -d ' ' -f 1
}

mkdir -p "$DIR"
while read -r name want; do
    f="$DIR/eigenworms_$name.pkl"
    if [ -f "$f" ]; then
        echo "$f exists, not downloaded"
    else
        curl -sSfL -o "$f.partial" "$URL/$COMMIT/$SRC/eigenworms_$name.pkl"
        mv "$f.partial" "$f"
        echo "downloaded $f"
    fi
    got=$(sha256 "$f")
    [ "$got" = "$want" ] || { echo "$f: SHA-256 $got, expected $want" >&2; exit 1; }
done <<'EOF'
train 435602a42e624f7d3a49dfe5959e65f0ec9980cc5947710d7b11201bb1df0bed
val 1666d98cef268a8a8e9c15e9d8fa9b7ba9a2c630ab86e18b330a2d2b4f3bfb38
test 137f564cc3fab69ce0917260be4ddf3e0b26547b1de221072acfafd936e64491
EOF
echo "EigenWorms: the three files of $DIR have the expected SHA-256"
