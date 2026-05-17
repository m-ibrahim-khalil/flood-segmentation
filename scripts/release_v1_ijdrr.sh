#!/usr/bin/env bash
#
# release_v1_ijdrr.sh -- guided GitHub-tag + Zenodo-DOI mint for the
# FloodLite IJDRR submission.
#
# Prerequisites (one-time, done outside this script):
#   1. Sign in to Zenodo (https://zenodo.org/) with your GitHub account.
#   2. Visit https://zenodo.org/account/settings/github/ and flip the
#      toggle next to "m-ibrahim-khalil/flood-segmentation" to ON.
#   3. (Optional) Add your ORCID at https://zenodo.org/account/settings/profile/
#      and fill it into the empty `orcid` field of .zenodo.json before
#      running this script.
#
# What this script does:
#   - Verifies the working tree is clean and on `main`.
#   - Verifies .zenodo.json and CITATION.cff parse and are not still
#     showing placeholder values.
#   - Verifies manuscript/main.pdf builds cleanly from the current source.
#   - Creates an annotated tag `v1.0-ijdrr` pointing at HEAD.
#   - Pushes the tag to origin (which triggers Zenodo's GitHub webhook
#     and starts the DOI mint).
#   - Prints the URLs you need to visit to (a) create the GitHub Release
#     from this tag, and (b) watch the Zenodo DOI appear.
#
# After this script finishes:
#   1. Visit https://github.com/m-ibrahim-khalil/flood-segmentation/releases/new?tag=v1.0-ijdrr
#      and click "Publish release" with the suggested release notes
#      (this script writes them to /tmp/floodlite_release_notes.md).
#   2. Within ~5 minutes the DOI will appear at
#      https://zenodo.org/account/settings/github/repository/m-ibrahim-khalil/flood-segmentation
#   3. Copy the DOI (form 10.5281/zenodo.NNNNNNN) into:
#         - manuscript/sections/09_appendix_reproducibility.tex
#         - manuscript/cover_letter.tex
#         - .zenodo.json (next release)
#         - CITATION.cff -> add "doi: ..." at the top level

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TAG_NAME="v1.0-ijdrr"
REMOTE="origin"
RELEASE_NOTES="/tmp/floodlite_release_notes.md"

cd "$REPO_ROOT"

echo "==> [1/6] Sanity-checking the working tree"
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "    ERROR: working tree is dirty. Commit or stash first."
  git status --short
  exit 1
fi
current_branch="$(git rev-parse --abbrev-ref HEAD)"
if [ "$current_branch" != "main" ]; then
  echo "    ERROR: you are on branch '$current_branch', not 'main'."
  echo "    Re-run from main once your release commit is on main."
  exit 1
fi
echo "    OK: clean working tree on main."

echo "==> [2/6] Sanity-checking .zenodo.json"
if [ ! -f .zenodo.json ]; then
  echo "    ERROR: .zenodo.json missing."
  exit 1
fi
python3 -c "import json,sys; json.load(open('.zenodo.json'))" \
  || { echo "    ERROR: .zenodo.json is not valid JSON."; exit 1; }
if grep -q '"orcid": ""' .zenodo.json; then
  echo "    WARN:  ORCID field in .zenodo.json is empty."
  echo "    You can add it later, but it will be missing from this release record."
  echo "    Continue anyway? [y/N]"
  read -r REPLY
  [[ "$REPLY" =~ ^[Yy]$ ]] || exit 1
fi
echo "    OK: .zenodo.json parses."

echo "==> [3/6] Sanity-checking CITATION.cff"
if [ ! -f CITATION.cff ]; then
  echo "    ERROR: CITATION.cff missing."
  exit 1
fi
echo "    OK: CITATION.cff present."

echo "==> [4/6] Verifying manuscript/main.pdf builds cleanly"
(cd manuscript && latexmk -pdf -interaction=nonstopmode main.tex > /tmp/floodlite_latex.log 2>&1) \
  || { echo "    ERROR: latexmk failed. See /tmp/floodlite_latex.log"; exit 1; }
echo "    OK: main.pdf builds."

echo "==> [5/6] Writing release notes to $RELEASE_NOTES"
cat > "$RELEASE_NOTES" <<'EOF'
## FloodLite v1.0-ijdrr -- IJDRR submission snapshot

This is the code-and-artifacts snapshot accompanying the manuscript
*FloodLite: A Multi-Platform Edge-Deployment Benchmark for Lightweight
Flood Segmentation*, under review at the International Journal of
Disaster Risk Reduction (IJDRR), 2026.

### What's in this release

- **Code** -- training and evaluation pipeline (`code/floodlite/`),
  benchmark scripts (`code/scripts/`), figure / table generators.
- **Pre-trained PyTorch checkpoints** -- teacher (UNet+EfficientNet-B0),
  MobileNetV2 baseline, three lightweight UNet students (MobileNetV3-S,
  EfficientNet-Lite0, MobileViT-XXS) under four KD configurations across
  three folds of FSSD.
- **ONNX exports** -- FP32 + static-QDQ INT8 for the three deployable
  students, per fold.
- **Per-fold metrics** under `code/runs/3fold/`.
- **LaTeX manuscript source** under `manuscript/`.

### Headline result

EfficientNet-Lite0-UNet INT8 runs at **58 FPS on a 2023 Apple M2 Pro
laptop CPU** and **6 FPS in a single-threaded browser WebAssembly
sandbox** at **3.83x size reduction** with **95% IoU recovery** relative
to the UNet+EfficientNet-B0 teacher.

### Important caveat on INT8 deployment

MobileNetV3-Small and MobileViT-XXS INT8 ONNX exports are included for
reproducibility but are **not recommended for deployment** -- they suffer
large post-training-quantization IoU drops (Delta IoU = -0.45) under the
static QDQ recipe. Use the EfficientNet-Lite0 INT8 ONNX (Delta = -0.043)
for production. See Section 5.3 of the manuscript for details.

### Reproducing

See `code/README.md` and `manuscript/sections/09_appendix_reproducibility.tex`.

### Citation

A Zenodo DOI will be minted automatically when this release is published.
Citation instructions appear in `CITATION.cff` at the repo root and on
the GitHub repo page.
EOF
echo "    OK: $RELEASE_NOTES"

echo "==> [6/6] Tagging and pushing"
if git rev-parse "$TAG_NAME" >/dev/null 2>&1; then
  echo "    WARN: tag $TAG_NAME already exists locally."
  echo "    Delete and recreate? [y/N]"
  read -r REPLY
  if [[ "$REPLY" =~ ^[Yy]$ ]]; then
    git tag -d "$TAG_NAME"
    git push --delete "$REMOTE" "$TAG_NAME" || true
  else
    echo "    Aborting."
    exit 1
  fi
fi
git tag -a "$TAG_NAME" -m "FloodLite v1.0 -- IJDRR submission snapshot"
git push "$REMOTE" "$TAG_NAME"
echo "    OK: pushed $TAG_NAME."

cat <<EOF

==================================================================
   Tag pushed. Now do the manual web steps:

   1. Create the GitHub Release:
        https://github.com/m-ibrahim-khalil/flood-segmentation/releases/new?tag=$TAG_NAME

      Paste the contents of $RELEASE_NOTES into the release description.
      Click "Publish release".

   2. The Zenodo GitHub webhook will fire automatically. Watch:
        https://zenodo.org/account/settings/github/repository/m-ibrahim-khalil/flood-segmentation

      The DOI will appear there within ~5 minutes.

   3. Copy the DOI (form 10.5281/zenodo.NNNNNNN) into:
        - manuscript/sections/09_appendix_reproducibility.tex
        - manuscript/cover_letter.tex
        - CITATION.cff  (add a top-level "doi: ..." field)

   4. Rebuild manuscript/main.pdf and re-commit the DOI substitution.
==================================================================
EOF
