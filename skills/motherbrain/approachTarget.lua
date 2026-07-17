function approachTarget(args)
    require(args.target ~= nil, "target is stale")
    approach(args.target, args.stop_range_m or 0.30)
    return {arrived = true}
end
