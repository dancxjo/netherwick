function driveDistanceSkill(args)
    require(args.distance_m ~= nil, "drive distance is required")
    driveDistance(args.distance_m, args.velocity_m_s)
    return {driven = true}
end
