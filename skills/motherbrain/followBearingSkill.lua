function followBearingSkill(args)
    require(args.bearing_rad ~= nil, "bearing is stale")
    followBearing(args.bearing_rad, 0.045)
    return {followed = true}
end
