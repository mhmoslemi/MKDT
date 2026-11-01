#!/bin/bash
set -euo pipefail
bench=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
# Compatibility entry point: Killarney now uses Lu.
exec bash "$bench/submit_mipa_cvpr_color_lu_killarney.sh" "$@"
