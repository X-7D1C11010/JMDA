#!/usr/bin/env bash

set -Eeuo pipefail

MODE="${1:-essential}"
DEST_ROOT="${2:-/mnt/data/lixiang/data/VDVRaw}"
ZENODO_ROOT="${ZENODO_ROOT:-https://zenodo.org}"

CLASSIFICATION_RECORD="14847258"
FULL_RECORD="13897485"
CLASSIFICATION_FILE="VDVRaw_classification_2categories.zip"
CLASSIFICATION_MD5="9e9cf7f87ad42696d3a23e21ec0ae8dc"
ANNOTATION_ZIP="annots_12bands.zip"
ANNOTATION_ZIP_MD5="b6c6f8d580cd9084e7783cd3d84262ab"
COCO_JSON="COCO_ASH_vessels_final_v02.json"
COCO_JSON_MD5="8d0c84d5be80f3279b5dbb0d6c7031e8"

usage() {
    cat <<'EOF'
Usage:
  bash download_vdvraw.sh classification [DEST]
  bash download_vdvraw.sh essential      [DEST]

Modes:
  classification  Download only the 38.9 MB two-class crop archive.
  essential       Download the classification archive, AIS annotations,
                  and 29 co-registered multispectral archives (~15-16 GB).

Default DEST:
  /mnt/data/lixiang/data/VDVRaw

Optional environment variable:
  ZENODO_ROOT     Override the Zenodo origin if your institution provides
                  an approved mirror or reverse proxy.
EOF
}

die() {
    echo "ERROR: $*" >&2
    exit 1
}

require_command() {
    command -v "$1" >/dev/null 2>&1 || die "Required command not found: $1"
}

case "$MODE" in
    classification|essential) ;;
    -h|--help)
        usage
        exit 0
        ;;
    *)
        usage >&2
        die "Unknown mode: $MODE"
        ;;
esac

require_command wget
require_command md5sum
require_command getent

check_zenodo_access() {
    local host resolved
    host="${ZENODO_ROOT#*://}"
    host="${host%%/*}"
    host="${host%%:*}"

    resolved="$(getent ahostsv4 "$host" 2>/dev/null | awk 'NR == 1 {print $1}')"
    if [[ -z "$resolved" ]]; then
        die "DNS lookup failed for ${host}. Check the server DNS or HTTPS proxy."
    fi
    if [[ "$resolved" == "0.0.0.0" || "$resolved" == "127."* ]]; then
        die "${host} resolves to ${resolved}. The server DNS/hosts policy is blocking Zenodo; wget cannot bypass that policy."
    fi

    echo "Zenodo endpoint: ${ZENODO_ROOT} (${resolved})"
}

download_file() {
    local url="$1"
    local output="$2"
    local expected_md5="${3:-}"

    mkdir -p "$(dirname "$output")"

    if [[ -f "$output" && -n "$expected_md5" ]]; then
        if echo "${expected_md5}  ${output}" | md5sum -c - >/dev/null 2>&1; then
            echo "Already verified: ${output}"
            return 0
        fi
    fi

    echo "Downloading: ${output}"
    wget \
        --continue \
        --retry-connrefused \
        --waitretry=5 \
        --timeout=60 \
        --tries=20 \
        --output-document="$output" \
        "$url"

    if [[ -n "$expected_md5" ]]; then
        echo "${expected_md5}  ${output}" | md5sum -c -
    fi
}

download_classification() {
    local output="${DEST_ROOT}/classification/${CLASSIFICATION_FILE}"
    local url="${ZENODO_ROOT}/records/${CLASSIFICATION_RECORD}/files/${CLASSIFICATION_FILE}?download=1"
    download_file "$url" "$output" "$CLASSIFICATION_MD5"
}

download_annotations() {
    local annotation_dir="${DEST_ROOT}/annotations"

    download_file \
        "${ZENODO_ROOT}/records/${FULL_RECORD}/files/${ANNOTATION_ZIP}?download=1" \
        "${annotation_dir}/${ANNOTATION_ZIP}" \
        "$ANNOTATION_ZIP_MD5"

    download_file \
        "${ZENODO_ROOT}/records/${FULL_RECORD}/files/${COCO_JSON}?download=1" \
        "${annotation_dir}/${COCO_JSON}" \
        "$COCO_JSON_MD5"
}

download_coregistered_archives() {
    local coreg_dir="${DEST_ROOT}/coreg"
    local index file url

    mkdir -p "$coreg_dir"
    for index in $(seq -w 1 29); do
        file="ASH_L0_CoReg_${index}.zip"
        url="${ZENODO_ROOT}/records/${FULL_RECORD}/files/${file}?download=1"
        download_file "$url" "${coreg_dir}/${file}"
    done
}

verify_zip_archives() {
    if ! command -v unzip >/dev/null 2>&1; then
        echo "WARNING: unzip is unavailable; skipping ZIP integrity checks." >&2
        return 0
    fi

    local archive
    while IFS= read -r -d '' archive; do
        echo "Testing ZIP: ${archive}"
        unzip -tq "$archive" >/dev/null
    done < <(find "$DEST_ROOT" -type f -name '*.zip' -print0)
}

check_zenodo_access
mkdir -p "$DEST_ROOT"
download_classification

if [[ "$MODE" == "essential" ]]; then
    download_annotations
    download_coregistered_archives
fi

verify_zip_archives

echo
echo "VDVRaw download completed."
echo "Destination: ${DEST_ROOT}"
du -sh "$DEST_ROOT"

