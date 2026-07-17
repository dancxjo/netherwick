function alignWithDockSkill(args)
    if charging() then
        stop()
        return {charging = true}
    end
    searchForDockSignal()
    alignWithDock()
    verifyCharging()
    return {charging = true}
end
