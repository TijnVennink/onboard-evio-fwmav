#!/usr/bin/env bash
# render history.csv as HiPlot and open in browser.
#   ./run_hiplot.sh                          # uses default history.csv
#   ./run_hiplot.sh optimizer_output_1800_000/history.csv
#   ./run_hiplot.sh --watch                  # auto-refresh every 30 s

set -euo pipefail

CSV="${1:-optimizer_1900_1_v11/history.csv}"
WATCH=false
INTERVAL=30
OUT="/tmp/hiplot_ekf.html"

# parse flags
for arg in "$@"; do
    case "$arg" in
        --watch) WATCH=true ;;
    esac
done

if [[ ! -f "$CSV" ]]; then
    echo "ERROR: CSV not found: $CSV" >&2
    exit 1
fi

render() {
    # activate venv if present and not already active
    if [[ -z "${VIRTUAL_ENV:-}" && -f ".venv/bin/activate" ]]; then
        source .venv/bin/activate
    fi

    echo "[hiplot] Rendering $CSV → $OUT"
    hiplot-render "$CSV" > "$OUT"
    echo "[hiplot] Done. $(wc -l < "$CSV") rows in CSV."
}

render
xdg-open "$OUT" 2>/dev/null || open "$OUT" 2>/dev/null || echo "Open manually: $OUT"

if $WATCH; then
    echo "[hiplot] Watch mode — refreshing every ${INTERVAL}s. Ctrl+C to stop."
    while true; do
        sleep "$INTERVAL"
        render
        echo "[hiplot] Refreshed at $(date +%H:%M:%S) — reload the browser tab."
    done
fi
