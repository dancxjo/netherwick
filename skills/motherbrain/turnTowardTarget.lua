function turnTowardTarget(args)
    require(args.bearing_rad ~= nil, "target bearing is stale")
    faceBearing(args.bearing_rad)
    return {faced = true}
end
