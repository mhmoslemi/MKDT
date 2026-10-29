#!/bin/bash
set -euo pipefail
: "${SLURM_JOB_ID:?Run through Slurm}"
data=${1:?Data root required}
folder="$data/FashionMNIST/raw"
mkdir -p "$folder"
cd "$folder"
# Official repository, with the archive checksums distributed by torchvision.
while read -r archive checksum; do
    if [[ ! -s $archive ]] || ! printf '%s  %s\n' "$checksum" "$archive" | md5sum -c --status; then
        curl --fail --location --retry 3 --connect-timeout 20 --max-time 120 \
            "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/$archive" \
            --output "$archive.download"
        printf '%s  %s\n' "$checksum" "$archive.download" | md5sum -c
        mv -- "$archive.download" "$archive"
    fi
    gzip -dc -- "$archive" > "${archive%.gz}.download"
    mv -- "${archive%.gz}.download" "${archive%.gz}"
    printf 'VERIFIED %s\n' "$folder/$archive"
done <<'FILES'
train-images-idx3-ubyte.gz 8d4fb7e6c68d591d4c3dfef9ec88bf0d
train-labels-idx1-ubyte.gz 25c81989df183df01b3e8a0aad5dffbe
t10k-images-idx3-ubyte.gz bef4ecab320f06d8554ea6380940ec79
t10k-labels-idx1-ubyte.gz bb300cfdad3c16e7a12a480ee83cd310
FILES
