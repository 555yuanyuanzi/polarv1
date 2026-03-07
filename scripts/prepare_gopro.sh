#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_ROOT="${1:-${ROOT_DIR}/data/gopro_v1}"
ARCHIVE_URL="${GOPRO_ARCHIVE_URL:-https://huggingface.co/datasets/snah/GOPRO_Large/resolve/main/GOPRO_Large.zip}"
ARCHIVE_SHA256="24532c036712515cf33803704421311c91f6a080e76049d7e5e7fd22389f128e"

DOWNLOAD_DIR="${TARGET_ROOT}/downloads"
RAW_DIR="${TARGET_ROOT}/raw"
FLAT_DIR="${TARGET_ROOT}/flat"
ARCHIVE_PATH="${DOWNLOAD_DIR}/GOPRO_Large.zip"

require_cmd() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "Missing required command: $1" >&2
        exit 1
    fi
}

resolve_abs_path() {
    local input_path="$1"
    if command -v realpath >/dev/null 2>&1; then
        realpath "${input_path}"
        return
    fi
    if command -v readlink >/dev/null 2>&1; then
        readlink -f "${input_path}"
        return
    fi
    python3 -c 'import os, sys; print(os.path.realpath(sys.argv[1]))' "${input_path}"
}

download_file() {
    local url="$1"
    local output="$2"
    if command -v curl >/dev/null 2>&1; then
        curl -L --fail --retry 3 --continue-at - -o "${output}" "${url}"
        return
    fi
    if command -v wget >/dev/null 2>&1; then
        wget -c -O "${output}" "${url}"
        return
    fi
    echo "Either curl or wget is required to download the dataset." >&2
    exit 1
}

resolve_source_root() {
    if [[ -d "${RAW_DIR}/GOPRO_Large/train" && -d "${RAW_DIR}/GOPRO_Large/test" ]]; then
        echo "${RAW_DIR}/GOPRO_Large"
        return
    fi
    if [[ -d "${RAW_DIR}/train" && -d "${RAW_DIR}/test" ]]; then
        echo "${RAW_DIR}"
        return
    fi
    echo "Could not find extracted GoPro train/test folders under ${RAW_DIR}" >&2
    exit 1
}

flatten_split() {
    local source_root="$1"
    local split="$2"
    local split_source="${source_root}/${split}"
    local blur_target="${FLAT_DIR}/${split}/blur"
    local sharp_target="${FLAT_DIR}/${split}/sharp"

    mkdir -p "${blur_target}" "${sharp_target}"
    find "${blur_target}" -maxdepth 1 -type l -delete
    find "${sharp_target}" -maxdepth 1 -type l -delete

    shopt -s nullglob
    local total_blur=0
    local total_sharp=0
    local scene_count=0

    for scene_dir in "${split_source}"/*; do
        [[ -d "${scene_dir}" ]] || continue
        [[ -d "${scene_dir}/blur" && -d "${scene_dir}/sharp" ]] || continue

        local scene_name
        scene_name="$(basename "${scene_dir}")"
        local scene_blur=0
        local scene_sharp=0
        scene_count=$((scene_count + 1))

        for image_path in "${scene_dir}/blur"/*; do
            [[ -f "${image_path}" ]] || continue
            local image_name
            image_name="$(basename "${image_path}")"
            ln -sfn "$(resolve_abs_path "${image_path}")" "${blur_target}/${scene_name}__${image_name}"
            scene_blur=$((scene_blur + 1))
            total_blur=$((total_blur + 1))
        done

        for image_path in "${scene_dir}/sharp"/*; do
            [[ -f "${image_path}" ]] || continue
            local image_name
            image_name="$(basename "${image_path}")"
            ln -sfn "$(resolve_abs_path "${image_path}")" "${sharp_target}/${scene_name}__${image_name}"
            scene_sharp=$((scene_sharp + 1))
            total_sharp=$((total_sharp + 1))
        done

        if [[ "${scene_blur}" -ne "${scene_sharp}" ]]; then
            echo "Pair count mismatch in ${scene_dir}: blur=${scene_blur}, sharp=${scene_sharp}" >&2
            exit 1
        fi
    done

    shopt -u nullglob

    if [[ "${total_blur}" -eq 0 || "${total_sharp}" -eq 0 ]]; then
        echo "No images found while flattening split=${split}." >&2
        exit 1
    fi
    if [[ "${total_blur}" -ne "${total_sharp}" ]]; then
        echo "Global pair count mismatch in split=${split}: blur=${total_blur}, sharp=${total_sharp}" >&2
        exit 1
    fi

    echo "Prepared ${split}: scenes=${scene_count}, pairs=${total_blur}"
}

main() {
    require_cmd unzip
    mkdir -p "${DOWNLOAD_DIR}" "${RAW_DIR}" "${FLAT_DIR}"

    if [[ ! -f "${ARCHIVE_PATH}" ]]; then
        echo "Downloading GOPRO_Large.zip to ${ARCHIVE_PATH}"
        download_file "${ARCHIVE_URL}" "${ARCHIVE_PATH}"
    else
        echo "Archive already exists: ${ARCHIVE_PATH}"
    fi

    if command -v sha256sum >/dev/null 2>&1; then
        local_hash="$(sha256sum "${ARCHIVE_PATH}" | awk '{print $1}')"
        if [[ "${local_hash}" != "${ARCHIVE_SHA256}" ]]; then
            echo "SHA256 mismatch for ${ARCHIVE_PATH}" >&2
            echo "Expected: ${ARCHIVE_SHA256}" >&2
            echo "Actual:   ${local_hash}" >&2
            exit 1
        fi
        echo "SHA256 verified."
    else
        echo "sha256sum not found; skipping checksum verification."
    fi

    if [[ ! -d "${RAW_DIR}/GOPRO_Large" && ! -d "${RAW_DIR}/train" ]]; then
        echo "Extracting archive to ${RAW_DIR}"
        unzip -q "${ARCHIVE_PATH}" -d "${RAW_DIR}"
    else
        echo "Extraction already present under ${RAW_DIR}"
    fi

    SOURCE_ROOT="$(resolve_source_root)"
    flatten_split "${SOURCE_ROOT}" "train"
    flatten_split "${SOURCE_ROOT}" "test"

    echo
    echo "GoPro dataset is ready for V1."
    echo "Set data.root_dir to:"
    echo "  ${FLAT_DIR}"
    echo
    echo "Expected V1 layout:"
    echo "  ${FLAT_DIR}/train/blur"
    echo "  ${FLAT_DIR}/train/sharp"
    echo "  ${FLAT_DIR}/test/blur"
    echo "  ${FLAT_DIR}/test/sharp"
}

main "$@"
