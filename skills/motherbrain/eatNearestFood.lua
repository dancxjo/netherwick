function eatNearestFood()
    local food = nearestVisible("food")
    require(food ~= nil, "no food is visible")
    reachForFood(food)
    bringToMouth(food)
    chew()
    swallow()
    return food
end
