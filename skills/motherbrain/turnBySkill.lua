function turnBySkill(args)
    require(args.angle_rad ~= nil, "turn angle is required")
    turnBy(args.angle_rad)
    return {turned = true}
end
