#!/bin/bash
set -euo pipefail
bench=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
export MIPA_CVPR_CLUSTER=killarney MIPA_CVPR_RUN=mipa_cvpr_color_crop_lr0p001_pool100
exec bash "$bench/submit_mipa_cvpr_color_vulcan.sh" "$@"
