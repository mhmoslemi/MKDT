#!/bin/bash
set -euo pipefail
bench=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
exec bash "$bench/submit_iic_both_wang_h100_killarney.sh" --lu-l40s "$@"
