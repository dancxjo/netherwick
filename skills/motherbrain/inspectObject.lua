function inspectObject(args)
    if args.target ~= nil then
        if args.bearing_rad ~= nil then
            faceBearing(args.bearing_rad)
        end
        observe(args.target)
    else
        turnBy(0.785398)
    end
    return {inspected = true}
end
