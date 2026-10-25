#!/system/bin/sh
# Launch separate neural stages to release each stage's GPU context and weights.
set -u
cd /data/local/tmp/mobile_tryon_20261024/generation_hybrid_low || exit 2
export LD_LIBRARY_PATH=..
start_seconds=$(date +%s)
printf 'start_epoch_seconds=%s\n' "$start_seconds" > generation.monitor.log
for phase in vae denoise decode; do
    ../mnn_generation generation.json "$phase" > "$phase.log" 2>&1 &
    bench_pid=$!
    printf 'phase=%s pid=%s\n' "$phase" "$bench_pid" >> generation.monitor.log
    while kill -0 "$bench_pid" 2>/dev/null; do
        available_kb=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
        printf 'phase=%s timestamp=%s mem_available_kb=%s\n' "$phase" "$(date +%s)" "$available_kb" >> generation.monitor.log
        awk '/VmHWM:|VmRSS:|VmSwap:/ {print}' "/proc/$bench_pid/status" >> generation.monitor.log 2>/dev/null
        if [ "$available_kb" -lt 131072 ]; then
            printf 'terminated_for_memory_pressure\n' >> generation.monitor.log
            kill -TERM "$bench_pid" 2>/dev/null
            break
        fi
        sleep 1
    done
    wait "$bench_pid"
    status=$?
    printf 'phase=%s exit_code=%s\n' "$phase" "$status" >> generation.monitor.log
    if [ "$status" -ne 0 ]; then exit "$status"; fi
done
end_seconds=$(date +%s)
printf 'end_epoch_seconds=%s online_wall_seconds=%s exit_code=0\n' "$end_seconds" "$((end_seconds-start_seconds))" >> generation.monitor.log
