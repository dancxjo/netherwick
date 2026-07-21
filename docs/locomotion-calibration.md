# Adaptive locomotion calibration

Motherbrain keeps an advisory locomotion-calibration state with conservative nominal distance
scales, rotation scales, and effective wheelbase. It does not rewrite Create odometry, motor
commands, speed limits, interlocks, or brainstem safety behavior. Until enough consistent evidence
exists, nominal parameters remain the operational fallback.

Straight episodes preserve reported and independently measured distance, direction, lateral drift,
endpoint heading error, environmental residual, repeat/loop support, confidence, surface, and tire
conditions. High-confidence repeated traversals update global scale slowly while retaining separate
left/right and forward/reverse estimates. Short, poorly aligned, low-confidence, or out-of-bounds
episodes remain in bounded history with explicit rejection reasons and do not update parameters.

Rotation episodes preserve commanded and wheel-odometry angles, trusted IMU angle, environmental
and loop angles, axle-center displacement, direction, confidence, and conditions. Trusted external
references are fused to update bounded effective-wheelbase and separate clockwise/counter-clockwise
scale estimates. Updates conflicting with established straight-line asymmetry are rejected.

Every estimate reports its safe bounds, uncertainty, and evidence count. Overall state reports
confidence, rejection counters, condition-tagged epochs, straight-line consistency, and retained
episode evidence. `Now.extensions["calibration.locomotion"]`, `WorldSnapshot`, and WorldLab
calibration assets identify the active epoch for replay and held-out validation.

Physical acceptance requires at least five measured straight runs of at least 2 m in each direction,
plus repeated slow 90, 180, and 360 degree rotations in both directions. Held-out runs must meet
declared distance, drift, heading, and axle-translation tolerances on the assembled robot.
