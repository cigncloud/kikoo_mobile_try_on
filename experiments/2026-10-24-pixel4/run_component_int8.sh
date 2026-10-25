#!/system/bin/sh
# Watch only this benchmark process; stop if physical available RAM falls below 128 MiB.
set -u
cd /data/local/tmp/mobile_tryon_20261024 || exit 2
manifest=$1
label=${manifest%.json}
./mnn_component_bench_streamed "$manifest" > "$label.log" 2>&1 &
bench_pid=$!
printf 'pid=%s\n' "$bench_pid" > "$label.monitor.log"
while kill -0 "$bench_pid" 2>/dev/null; do
    available_kb=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
    printf 'timestamp=%s mem_available_kb=%s\n' "$(date +%s)" "$available_kb" >> "$label.monitor.log"
    awk '/VmHWM:|VmRSS:|VmSwap:/ {print}' "/proc/$bench_pid/status" >> "$label.monitor.log" 2>/dev/null
    if [ "$available_kb" -lt 131072 ]; then
        printf 'terminated_for_memory_pressure\n' >> "$label.monitor.log"
        dumpsys meminfo "$bench_pid" > "$label.memory_pressure.txt"
        kill -TERM "$bench_pid" 2>/dev/null
        break
    fi
    sleep 1
done
wait "$bench_pid"
bench_status=$?
printf 'exit_code=%s\n' "$bench_status" >> "$label.monitor.log"
exit "$bench_status"
