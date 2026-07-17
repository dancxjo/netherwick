function driveFor(args)
    -- Escape-danger affordances are intentionally brief (normally 300 ms).
    -- Complete one observed act comfortably inside that invocation deadline;
    -- the conductor may then select the turn/probe phase from the new Now.
    local duration_ms = math.min(100, math.max(50, (args.maximum_duration_ms or 250) / 2))
    drive(-0.10, duration_ms)
    return {duration_ms = duration_ms}
end
