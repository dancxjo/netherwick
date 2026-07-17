function retreatFromCliff(args)
    if cliffActive() then
        carefully("cliff", function()
            retreat(100)
        end)
    end
    completeHazardRecovery("cliff")
    require(cliffIsClear(), "cliff remains active after bounded retreat")
    return {clear = true}
end
