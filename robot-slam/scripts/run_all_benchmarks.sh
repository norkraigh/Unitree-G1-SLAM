#!/usr/bin/env bash
set -euo pipefail

ROOT="$HOME/Documents/TFM-Miquel_Reynes/Follower_Project/robot-internal-files/robot-slam"
cd "$ROOT"

CONFIG="config/livox_slam_config.json"
OUTPUT_DIR="results/final_10x"
MASTER_CSV="$OUTPUT_DIR/tfm_slam_benchmark.csv"

BAGS=("00" "01" "02" "03" "04" "05" "06" "07")
ALGORITHMS=("icp" "kiss_icp" "fast_lio2")
REPETITIONS=10

# ------------------------------------------------------------
# ROS environment
# ------------------------------------------------------------
# Run this script from the g1-slam-pc conda environment.
set +u
source /opt/ros/humble/setup.bash
source install/setup.bash
set -u

export ROS_DOMAIN_ID=10
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export PYTHONNOUSERSITE=1

# ------------------------------------------------------------
# Preflight
# ------------------------------------------------------------
if [[ ! -f "$CONFIG" ]]; then
    echo "ERROR: Config not found: $CONFIG"
    exit 1
fi

ros2 pkg prefix kiss_icp >/dev/null 2>&1 || {
    echo "ERROR: kiss_icp is not available in the sourced workspace."
    exit 1
}

ros2 pkg prefix fast_lio >/dev/null 2>&1 || {
    echo "ERROR: fast_lio is not available in the sourced workspace."
    exit 1
}

for BAG in "${BAGS[@]}"; do
    BAG_PATH="../../bags/tfm-bag-${BAG}"
    if [[ ! -d "$BAG_PATH" ]]; then
        echo "ERROR: Missing bag: $BAG_PATH"
        exit 1
    fi
done

# Avoid mixing final measurements with development runs.
if [[ -f "$MASTER_CSV" ]]; then
    echo "ERROR: $MASTER_CSV already exists."
    echo "Move/delete results/final_10x before starting a new final campaign."
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

BACKUP="$(mktemp "${CONFIG}.benchmark_backup.XXXXXX")"
cp "$CONFIG" "$BACKUP"

restore_config() {
    local exit_code=$?
    echo
    echo "Restoring original configuration..."
    if [[ -f "$BACKUP" ]]; then
        cp "$BACKUP" "$CONFIG"
        rm -f "$BACKUP"
    fi
    exit "$exit_code"
}
trap restore_config EXIT INT TERM

set_algorithm() {
    local algorithm="$1"

    python - "$CONFIG" "$algorithm" <<'PY'
import json
import sys

path = sys.argv[1]
algorithm = sys.argv[2]

with open(path, "r", encoding="utf-8") as f:
    config = json.load(f)

config.setdefault("slam", {})["algorithm"] = algorithm

with open(path, "w", encoding="utf-8") as f:
    json.dump(config, f, indent=4, ensure_ascii=False)

print(f"Configured slam.algorithm = {algorithm}")
PY
}

# ------------------------------------------------------------
# Final benchmark campaign
# ------------------------------------------------------------
#
# Fixed execution order:
#
# tfm-bag-00:
#   repetition 1: ICP -> KISS-ICP -> FAST-LIO2
#   repetition 2: ICP -> KISS-ICP -> FAST-LIO2
#   ...
#   repetition 10: ICP -> KISS-ICP -> FAST-LIO2
#
# Then tfm-bag-01, tfm-bag-02, ..., tfm-bag-07.
#
TOTAL_RUNS=$(( ${#BAGS[@]} * ${#ALGORITHMS[@]} * REPETITIONS ))
CURRENT_RUN=0

echo "======================================================"
echo " FINAL TFM BENCHMARK"
echo " 8 bags x 3 algorithms x 10 repetitions = $TOTAL_RUNS runs"
echo " Order: complete bag-00, then bag-01, ..., bag-07"
echo " Per repetition: ICP -> KISS-ICP -> FAST-LIO2"
echo " Output: $OUTPUT_DIR"
echo " Playback: LiDAR + IMU only, rate 1.0"
echo "======================================================"

for BAG in "${BAGS[@]}"; do

    BAG_PATH="../../bags/tfm-bag-${BAG}"

    echo
    echo "######################################################"
    echo " STARTING tfm-bag-$BAG"
    echo "######################################################"

    for RUN in $(seq 1 "$REPETITIONS"); do

        for ALGORITHM in "${ALGORITHMS[@]}"; do

            CURRENT_RUN=$((CURRENT_RUN + 1))

            echo
            echo "======================================================"
            echo " Campaign  : $CURRENT_RUN / $TOTAL_RUNS"
            echo " Bag       : tfm-bag-$BAG"
            echo " Repetition: $RUN / $REPETITIONS"
            echo " Algorithm : $ALGORITHM"
            echo "======================================================"

            set_algorithm "$ALGORITHM"

            python scripts/run_bag_benchmark.py \
                "$BAG_PATH" \
                --config "$CONFIG" \
                --output-dir "$OUTPUT_DIR" \
                --rate 1.0 \
                --startup-delay 6.0 \
                --drain-delay 5.0 \
                --topics \
                    /livox/lidar \
                    /livox/imu

            # Give ROS/DDS and child processes a short wall-clock interval
            # before starting the next independent run.
            sleep 2
        done
    done

    echo
    echo "######################################################"
    echo " COMPLETED tfm-bag-$BAG"
    echo "######################################################"

done

# ------------------------------------------------------------
# Validate expected number of rows
# ------------------------------------------------------------
python - "$MASTER_CSV" "$REPETITIONS" <<'PY'
import csv
import sys
from collections import Counter

path = sys.argv[1]
expected = int(sys.argv[2])

algorithms = ("icp", "kiss_icp", "fast_lio2")
bags = tuple(f"tfm-bag-{i:02d}" for i in range(8))
modes = ("raw", "optimized")

with open(path, newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

counts = Counter(
    (r.get("algorithm"), r.get("bag"), r.get("result_mode"))
    for r in rows
)

invalid_rows = []
for r in rows:
    try:
        raw_samples = int(float(r.get("raw_odom_samples") or 0))
    except ValueError:
        raw_samples = 0
    try:
        status_samples = int(float(r.get("status_samples") or 0))
    except ValueError:
        status_samples = 0

    optimized_available = str(
        r.get("optimized_path_available", "")
    ).strip().lower() in ("true", "1", "yes")

    if raw_samples <= 0 or status_samples <= 0 or not optimized_available:
        invalid_rows.append(
            (
                r.get("run_id"),
                r.get("algorithm"),
                r.get("bag"),
                r.get("result_mode"),
                raw_samples,
                status_samples,
                optimized_available,
            )
        )

print()
print("=" * 72)
print("FINAL ROW COUNT VALIDATION")
print("=" * 72)

ok = True
for bag in bags:
    for algorithm in algorithms:
        raw = counts[(algorithm, bag, "raw")]
        opt = counts[(algorithm, bag, "optimized")]
        print(
            f"{bag:12s}  {algorithm:10s}  "
            f"RAW={raw:2d}  OPTIMIZED={opt:2d}"
        )
        if raw != expected or opt != expected:
            ok = False

expected_rows = len(algorithms) * len(bags) * expected * len(modes)

print("-" * 72)
print(f"CSV rows: {len(rows)} / expected {expected_rows}")
print(f"Invalid/empty rows: {len(invalid_rows)}")

if invalid_rows:
    print()
    print("Invalid runs:")
    for item in invalid_rows:
        print(
            "  run_id=%s algorithm=%s bag=%s mode=%s "
            "raw_samples=%s status_samples=%s optimized_available=%s"
            % item
        )

if not ok or len(rows) != expected_rows or invalid_rows:
    raise SystemExit(
        "ERROR: Final benchmark is incomplete or contains empty/invalid runs."
    )

print("OK: benchmark campaign is complete.")
PY

echo
echo "======================================================"
echo " BENCHMARK COMPLETED SUCCESSFULLY"
echo "======================================================"
echo "Master CSV:"
echo "$ROOT/$MASTER_CSV"
