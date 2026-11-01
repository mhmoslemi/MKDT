#!/bin/bash
set -euo pipefail
bench=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
export MIPA_CVPR_CLUSTER=vulcan MIPA_CVPR_RUN=mipa_cvpr_color_crop
exec bash "$bench/submit_mipa_cvpr_color_vulcan.sh" "$@"
