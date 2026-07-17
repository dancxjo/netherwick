function reachForFood(food)
    together(
        function()
            approach(food, 0.20)
        end,
        function()
            lookAt(food)
        end,
        function()
            say("I found food.")
        end
    )
    grasp(food)
    return food
end
