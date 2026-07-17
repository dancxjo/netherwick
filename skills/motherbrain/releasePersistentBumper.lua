function releasePersistentBumper(args)
    if contactActive() then
        carefully("bumper_front", function()
            retreat(100)
        end)
    end
    completeHazardRecovery("bumper_front")
    require(not contactActive(), "bumper remains active after bounded retreat")
    return {clear = true}
end
