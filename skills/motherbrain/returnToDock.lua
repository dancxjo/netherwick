function returnToDock(args)
    local dock = searchForDock()
    require(dock ~= nil, "dock signal disappeared")
    together(
        function()
            approach(dock, 0.35)
        end,
        function()
            lookAt(dock)
        end,
        function()
            say("I am returning to Home Base.")
        end
    )
    alignWithDock()
    stop()
    return verifyCharging()
end
