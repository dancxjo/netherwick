"""
create1_cargo_bay_adapter.py

A first-pass build123d model for an iRobot Create 1 cargo-bay adapter/enclosure base.

The numbers below are hand-extracted from the AutonomyLab Create 1 COLLADA mesh.
The mesh's coordinates behave like inches even though the COLLADA metadata says meters;
we scale them using the URDF/Create diameter: 12.99016 mesh units ≈ 330.2 mm.

Print this as a fit coupon first. Do not trust any latch/snap until plastic says yes.
"""

from build123d import *

# -----------------------------
# Mesh-derived scale and helpers
# -----------------------------
MM_PER_MESH_UNIT = 330.2 / 12.99016  # ≈25.41924 mm per mesh unit

def U(v: float) -> float:
    """Create mesh units to millimeters."""
    return v * MM_PER_MESH_UNIT

# -----------------------------
# Mesh-derived cargo bay numbers
# -----------------------------
# Main cargo bay rectangular span from the Create 1 mesh:
# x: -4.70027 .. 0.44711, y: -3.81614 .. +3.81614, z floor: about -0.929
BAY_X_BACK = U(-4.70027)
BAY_X_FRONT = U(0.44711)
BAY_Y_HALF = U(3.81614)
BAY_LENGTH = BAY_X_FRONT - BAY_X_BACK       # ≈130.85 mm
BAY_WIDTH = 2 * BAY_Y_HALF                  # ≈194.01 mm

# Smaller centered connector/plinth rectangle from the mesh:
# x: -0.23289 .. 0.44711, y: -1.19750 .. +1.19750, z: -0.149 .. 0.051
PLINTH_X_BACK = U(-0.23289)
PLINTH_X_FRONT = U(0.44711)
PLINTH_Y_HALF = U(1.19750)
PLINTH_LENGTH = PLINTH_X_FRONT - PLINTH_X_BACK   # ≈17.28 mm
PLINTH_WIDTH = 2 * PLINTH_Y_HALF                 # ≈60.87 mm
PLINTH_HEIGHT = U(0.051 - (-0.149))              # ≈5.08 mm

# The mesh suggests the bay bottom is far below the top-shell reference.
# For an adapter socket we only need a shallow capture, not the whole robot cavity.
CLEARANCE = 0.55          # general printer clearance, mm
WALL = 2.4
BASE_THICKNESS = 2.8
LIP_HEIGHT = 7.5
FLOOR_CAPTURE_DEPTH = 2.0

# DSUB-25 cutout guess, intentionally generous.
# This mates to the visible connector/plinth zone, not a manufactured DSUB spec model.
DSUB_BODY_W = 54.5
DSUB_BODY_D = 12.0
DSUB_BODY_H = 13.2
DSUB_SCREW_SPACING = 47.0
DSUB_SCREW_DIA = 3.2

# Board slot defaults, tune after choosing actual board positions.
PCB_THICKNESS = 1.6
SLOT_W = PCB_THICKNESS + 0.55
SLOT_H = 8.5
SLOT_D = 24.0

# -----------------------------
# Model
# -----------------------------
# Coordinate convention for this printed part:
# X = bay length, positive toward Create connector/front.
# Y = bay width, left/right.
# Z = vertical.

outer_len = BAY_LENGTH + 2 * WALL
outer_w = BAY_WIDTH + 2 * WALL
outer_h = BASE_THICKNESS + LIP_HEIGHT

with BuildPart() as part:
    # Bathtub body
    Box(outer_len, outer_w, outer_h, align=(Align.CENTER, Align.CENTER, Align.MIN))

    # Main cargo-bay pocket. This is the actual Create-facing negative.
    with Locations((0, 0, BASE_THICKNESS)):
        Box(
            BAY_LENGTH + CLEARANCE,
            BAY_WIDTH + CLEARANCE,
            LIP_HEIGHT + 0.2,
            mode=Mode.SUBTRACT,
            align=(Align.CENTER, Align.CENTER, Align.MIN),
        )

    # A small lower relief so the socket does not bottom out on any molded waviness.
    with Locations((0, 0, BASE_THICKNESS - FLOOR_CAPTURE_DEPTH)):
        Box(
            BAY_LENGTH - 8.0,
            BAY_WIDTH - 8.0,
            FLOOR_CAPTURE_DEPTH + 0.2,
            mode=Mode.SUBTRACT,
            align=(Align.CENTER, Align.CENTER, Align.MIN),
        )

    # Front centered plinth / connector keepout, from the DAE small rectangle.
    # Position is derived relative to the bay center.
    bay_center_x = (BAY_X_BACK + BAY_X_FRONT) / 2
    plinth_center_x = (PLINTH_X_BACK + PLINTH_X_FRONT) / 2
    plinth_x_local = plinth_center_x - bay_center_x

    with Locations((plinth_x_local, 0, BASE_THICKNESS - 0.05)):
        Box(
            PLINTH_LENGTH + 2 * CLEARANCE,
            PLINTH_WIDTH + 2 * CLEARANCE,
            PLINTH_HEIGHT + 1.0,
            mode=Mode.SUBTRACT,
            align=(Align.CENTER, Align.CENTER, Align.MIN),
        )

    # DSUB-25 throat through the front wall, centered on the plinth.
    # This is deliberately a little taller/wider than a perfect DSUB so the connector locks mechanically
    # without forcing the board to become a structural member.
    front_wall_y = 0
    with Locations((outer_len / 2 - WALL / 2, front_wall_y, BASE_THICKNESS + DSUB_BODY_H / 2)):
        Box(
            WALL + 1.0,
            DSUB_BODY_W + CLEARANCE,
            DSUB_BODY_H + CLEARANCE,
            mode=Mode.SUBTRACT,
            align=(Align.CENTER, Align.CENTER, Align.CENTER),
        )

    # DSUB screw holes through front face.
    for y in (-DSUB_SCREW_SPACING / 2, DSUB_SCREW_SPACING / 2):
        with Locations((outer_len / 2 - WALL / 2, y, BASE_THICKNESS + DSUB_BODY_H / 2)):
            Cylinder(
                radius=DSUB_SCREW_DIA / 2,
                height=WALL + 2.0,
                rotation=(0, 90, 0),
                mode=Mode.SUBTRACT,
            )

    # Two low PCB slots inside the tub.
    # These are placeholders for TXS0108E / IMU boards: slide-in slots, not shelves.
    for y in (-36.0, 36.0):
        with Locations((-10.0, y, BASE_THICKNESS)):
            Box(SLOT_D, SLOT_W, SLOT_H, mode=Mode.SUBTRACT, align=(Align.CENTER, Align.CENTER, Align.MIN))

    # Pin nubs for board mounting holes. Trim or move after measuring your specific breakouts.
    for y in (-36.0, 36.0):
        for x in (-18.0, -2.0):
            with Locations((x, y, BASE_THICKNESS)):
                Cylinder(radius=1.05, height=2.2, mode=Mode.ADD)

    # Friendly outside corners, because hands are also sensors. Keep this away from
    # tiny vertical edges created by connector and relief cutouts.
    outer_vertical_edges = [
        edge
        for edge in part.edges().filter_by(Axis.Z)
        if abs(abs(edge.center().X) - outer_len / 2) < 0.01
        and abs(abs(edge.center().Y) - outer_w / 2) < 0.01
    ]
    fillet(outer_vertical_edges, radius=1.0)

if "show_object" in globals():
    show_object(part.part, name="create1_cargo_bay_adapter")
export_stl(part.part, "create1_cargo_bay_adapter.stl")
export_step(part.part, "create1_cargo_bay_adapter.step")

print("Mesh-derived dimensions:")
print(f"  BAY_LENGTH  = {BAY_LENGTH:.2f} mm")
print(f"  BAY_WIDTH   = {BAY_WIDTH:.2f} mm")
print(f"  PLINTH_LEN  = {PLINTH_LENGTH:.2f} mm")
print(f"  PLINTH_W    = {PLINTH_WIDTH:.2f} mm")
print(f"  PLINTH_H    = {PLINTH_HEIGHT:.2f} mm")
