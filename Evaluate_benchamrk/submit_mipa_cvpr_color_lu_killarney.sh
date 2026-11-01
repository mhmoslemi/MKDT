#!/bin/bash
set -euo pipefail
bench=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
export MIPA_CVPR_CLUSTER=killarney
exec bash "$bench/submit_mipa_cvpr_color_vulcan.sh" "$@"
