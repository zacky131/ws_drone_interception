#!/usr/bin/env bash
set -euo pipefail

NUC_WS=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
failures=0
CHECKSUM_MANIFEST="$NUC_WS/deployment/nuc-bundle-files.sha256"
ALLOW_GENERATED=0

if [[ ${1:-} == "--allow-generated" ]]; then
  ALLOW_GENERATED=1
  shift
fi
if [[ $# -ne 0 ]]; then
  echo "usage: $0 [--allow-generated]" >&2
  exit 2
fi

pass() { echo "PASS  $*"; }
fail() { echo "FAIL  $*" >&2; failures=$((failures + 1)); }

if [[ ! -f "$CHECKSUM_MANIFEST" ]]; then
  fail "missing deployment/nuc-bundle-files.sha256"
elif grep -Eq '(^|/)(results|data|build|install|log)/' "$CHECKSUM_MANIFEST"; then
  fail "checksum inventory references an excluded large-output directory"
elif (cd "$NUC_WS" && sha256sum --check --quiet --strict deployment/nuc-bundle-files.sha256); then
  listed_files=$(wc -l < "$CHECKSUM_MANIFEST")
  pass "all $listed_files inventoried bundle files match"
else
  fail "one or more inventoried files are missing or changed"
fi

if [[ $ALLOW_GENERATED -eq 0 ]]; then
  for excluded in results data build install log; do
    if [[ -e "$NUC_WS/$excluded" ]]; then
      fail "transfer bundle unexpectedly contains $excluded"
    else
      pass "excluded $excluded"
    fi
  done

  if find "$NUC_WS" -type f \
    \( -name '*.pyc' -o -name '*.so' \) -print -quit | grep -q .; then
    fail "transfer bundle contains generated Python or binary artifacts"
  else
    pass "excluded generated Python and binary artifacts"
  fi
else
  pass "generated build output allowed; inventoried source still verified"
fi

if [[ $failures -ne 0 ]]; then
  echo "NUC BUNDLE VERIFICATION FAILED ($failures checks)" >&2
  exit 1
fi
echo "NUC BUNDLE VERIFICATION PASS"
