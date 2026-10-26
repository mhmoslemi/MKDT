#!/bin/bash
set -euo pipefail
export TABLE_ACCOUNT=aip-yiweilu
exec bash "$(dirname -- "${BASH_SOURCE[0]}")/submit_remaining_tables_half_vulcan.sh" "$@"
