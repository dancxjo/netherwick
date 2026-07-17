function greet(args)
    local person = args.target
    require(person ~= nil, "greet requires a current person")

    together(
        function()
            face(person)
        end,
        function()
            if person.name then
                say("Hello " .. person.name .. ".")
            else
                say("Hello.")
            end
        end
    )

    acknowledge(person)
    reportProgress("social_acknowledgment", 1.0)
    return {
        person_id = person.id,
        name = person.name,
        acknowledged = true,
    }
end
