#!/usr/bin/env bash
# Zenodo deposit helper. Assumes ZENODO_TOKEN is set in the environment.
#
# Workflow:
#   1. Update CITATION.cff version + date
#   2. git tag and push
#   3. POST to Zenodo API with the release tarball
#   4. Print the returned DOI for inclusion in the AMF cover letter
#
# Usage:
#   ZENODO_TOKEN=xxx bash scripts/zenodo_release.sh 1.0.1
#
# Dry run (no API calls) if ZENODO_TOKEN is unset.

set -euo pipefail
cd "$(dirname "$0")/.."

VERSION="${1:?version required, e.g. 0.2.0}"
DATE="$(date -u +%Y-%m-%d)"

# 1. Update CITATION.cff
python3 - <<PY
import re, pathlib
p = pathlib.Path("CITATION.cff")
text = p.read_text()
text = re.sub(r"^version:.*$", "version: \"$VERSION\"", text, flags=re.M)
text = re.sub(r'^date-released:.*$', "date-released: \"$DATE\"", text, flags=re.M)
p.write_text(text)
print("Updated CITATION.cff")
PY

# 2. Tag
git tag -a "v$VERSION" -m "AMF revision release $VERSION"
echo "Created tag v$VERSION (run 'git push --tags' to publish)"

# 3. Build sdist
mkdir -p dist
TARBALL="dist/amf-cjp-liquidation-benchmark-$VERSION.tar.gz"
git archive --format=tar.gz --prefix="amf-cjp-liquidation-benchmark-$VERSION/" \
    -o "$TARBALL" HEAD
echo "Wrote $TARBALL"

# 4. Zenodo POST
if [[ -z "${ZENODO_TOKEN:-}" ]]; then
    echo "ZENODO_TOKEN not set — dry run. Tarball is at $TARBALL."
    exit 0
fi

ZENODO_URL="${ZENODO_URL:-https://zenodo.org}"
echo "Creating Zenodo deposit at $ZENODO_URL ..."

DEPOSIT=$(curl -s -X POST "$ZENODO_URL/api/deposit/depositions" \
    -H "Authorization: Bearer $ZENODO_TOKEN" \
    -H "Content-Type: application/json" \
    -d '{"metadata":{"title":"Sample Complexity of Calibration versus Model-Free Learning in CJP Optimal Liquidation","upload_type":"software"}}')
DEPOSIT_ID=$(echo "$DEPOSIT" | python3 -c "import json,sys; print(json.load(sys.stdin)['id'])")
echo "Deposit ID: $DEPOSIT_ID"

curl -s -X POST "$ZENODO_URL/api/deposit/depositions/$DEPOSIT_ID/files" \
    -H "Authorization: Bearer $ZENODO_TOKEN" \
    -F "name=$(basename $TARBALL)" \
    -F "file=@$TARBALL"

echo
echo "Visit $ZENODO_URL/deposit/$DEPOSIT_ID to publish the deposit and obtain the DOI."
