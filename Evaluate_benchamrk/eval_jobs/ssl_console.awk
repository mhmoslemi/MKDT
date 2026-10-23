# Keep full logs upstream; throttle only the terminal's SSL progress.
/^SSL CHECK / { interval = 0 }
/^SSL .*: \{/ && match($0, /"steps":[[:space:]]*[0-9]+/) {
    steps = substr($0, RSTART, RLENGTH)
    gsub(/[^0-9]/, "", steps)
    bucket = int(steps / 200)
    if (match($0, /"lr":[[:space:]]*[^,}]+/)) {
        lr = substr($0, RSTART, RLENGTH)
        sub(/^.*:[[:space:]]*/, "", lr)
        if (bucket <= interval && lr + 0 != 0)
            next
    }
    interval = bucket
    gsub(/,[[:space:]]*"projector_std":[[:space:]]*[^,}]+/, "")
    gsub(/,[[:space:]]*"seconds":[[:space:]]*[^,}]+/, "")
}
{ print; fflush() }
