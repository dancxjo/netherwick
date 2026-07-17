function holdHeadingSkill(args)
    require(args.bearing_rad ~= nil, "heading error is stale")
    holdHeading(args.bearing_rad, 0.0)
    return {held = true}
end
