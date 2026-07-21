function alignWithDockSkill(args)
    if charging() then
        stop()
        return verifyCharging()
    end
    searchForDockSignal()
    alignWithDock()
    return verifyCharging()
end
