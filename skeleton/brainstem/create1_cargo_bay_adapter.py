"""
create1_brainstem_row0_box_v4.py

A first tangible build123d model for the Netherwick / Pete brainstem enclosure.

Architecture:
- Full-width Row-0 command-module-style box sitting on the Create 1 shelf.
- Plinth / cargo-bay D-sub connector lives in row 0.0, butted against the north wall.
- Enclosure body occupies row 0.1, with a rectangular north-center bite around the plinth.
- Bottom is a hollow open-top box; top is a snap lid with a small front lip.
- MPU-6050 hangs from the lid on the west/left side of the connector.
- Pico W hangs from the lid on the east/right side of the connector, USB facing north.
- TXS0108E level shifter sits belly-up on the floor in a snap dock.
- Connector seating/extraction yoke ribs route push/pull loads toward D-sub shell/flanges.

This is a fit/design model, not a final measured part. The constants at top are meant
for fast iteration after one small print or FreeCAD inspection.
"""

from build123d import *

# ------------------------------------------------------------
# Coordinate system
# X: west/east across row 0. Centered on connector/plinth.
# Y: south/north. South/front/open cargo bay is negative Y. North/hinge/robot center is positive Y.
# Z: up from shelf top.
# ------------------------------------------------------------

# Overall row-0 box dimensions. These are deliberately conservative placeholders.
BOX_W = 194.0          # full cargo bay / shelf width, mm-ish from mesh-derived width
BOX_D = 42.0           # row-0 usable shelf depth south of plinth strip
BOX_H = 18.0           # bottom box wall height
WALL = 2.2
FLOOR = 1.8
CLEAR = 0.45

# Plinth / D-sub north bite. The plinth sits along the north edge, centered in X.
PLINTH_W = 64.0        # includes clearance around raised plastic plinth
PLINTH_D = 18.0        # how far south the bite intrudes from north wall
PLINTH_CLEAR_H = 12.0  # clearance height under/around connector shell

# D-sub male connector shell/flange approximation, used for clearance/yoke pads.
DSUB_W = 56.0
DSUB_D = 13.5
DSUB_H = 13.0
DSUB_Z = FLOOR + 4.5   # approximate shell center height inside box
DSUB_Y = BOX_D / 2 - PLINTH_D + 7.0

# Screw holes at west/east ends, left open for now.
SCREW_X = 82.0
SCREW_Y = -5.0
SCREW_R = 2.0
SCREW_BOSS_R = 5.5

# Lid parameters
LID_THICK = 2.0
LID_OVERLAP = 1.2      # lid pockets down inside bottom box
LID_CLEAR = 0.25
LIP_W = 42.0
LIP_D = 5.0
LIP_H = 3.0

# Board guesses. Adjust to your exact breakouts.
MPU_W = 21.0
MPU_D = 16.0
MPU_HOLE_SPAN_Y = 10.5
MPU_HOLE_X_FROM_CENTER = 7.0
MPU_X = -38.0
MPU_Y = -7.0
MPU_POST_R = 1.25
MPU_POST_H = 7.0
MPU_LED_WINDOW_R = 2.0

PICO_W = 51.0
PICO_D = 21.0
PICO_X = 43.0
PICO_Y = -8.0
PICO_POST_R = 1.25
PICO_POST_H = 7.0
USB_EXIT_W = 14.0
USB_EXIT_D = 8.0

TXS_W = 33.0
TXS_D = 16.0
TXS_X = 0.0
TXS_Y = -18.0
TXS_DOCK_WALL = 1.2
TXS_DOCK_H = 3.0

# Yoke ribs / contact pads. These are conceptual and intentionally visible.
YOKE_PAD_W = 8.0
YOKE_PAD_D = 6.0
YOKE_PAD_H = 2.0
YOKE_X = DSUB_W / 2 + 5.0
YOKE_RIB_T = 2.0


def safe_fillet(edges, radius):
    try:
        fillet(edges, radius=radius)
    except Exception as exc:
        print(f"Skipping fillet radius={radius}: {exc}")


def make_bottom_box():
    """Bottom open-top hollow box with north-center plinth bite."""
    with BuildPart() as p:
        # Main solid slab/walls envelope, aligned so north edge is +Y.
        Box(BOX_W, BOX_D, BOX_H, align=(Align.CENTER, Align.CENTER, Align.MIN))

        # Hollow interior. Leaves floor and walls.
        with Locations((0, 0, FLOOR)):
            Box(
                BOX_W - 2 * WALL,
                BOX_D - 2 * WALL,
                BOX_H,
                mode=Mode.SUBTRACT,
                align=(Align.CENTER, Align.CENTER, Align.MIN),
            )

        # North-center plinth bite through the north wall and body.
        # This represents row 0.0: plinth/connector strip butted against north wall.
        with Locations((0, BOX_D / 2 - PLINTH_D / 2 + CLEAR, 0)):
            Box(
                PLINTH_W + 2 * CLEAR,
                PLINTH_D + 2 * CLEAR,
                PLINTH_CLEAR_H,
                mode=Mode.SUBTRACT,
                align=(Align.CENTER, Align.CENTER, Align.MIN),
            )

        # Larger upper clearance around D-sub shell in same bite.
        with Locations((0, DSUB_Y, DSUB_Z)):
            Box(
                DSUB_W + 2 * CLEAR,
                DSUB_D + 2 * CLEAR,
                DSUB_H + 2 * CLEAR,
                mode=Mode.SUBTRACT,
                align=(Align.CENTER, Align.CENTER, Align.CENTER),
            )

        # Screw bosses with through holes, at west/east ends.
        for sx in (-SCREW_X, SCREW_X):
            with Locations((sx, SCREW_Y, FLOOR)):
                Cylinder(SCREW_BOSS_R, BOX_H - FLOOR, mode=Mode.ADD, align=(Align.CENTER, Align.CENTER, Align.MIN))
            with Locations((sx, SCREW_Y, -0.1)):
                Cylinder(SCREW_R, BOX_H + 0.2, mode=Mode.SUBTRACT, align=(Align.CENTER, Align.CENTER, Align.MIN))

        # TXS0108E belly-up snap dock on the floor: four low retaining rails.
        dock_outer_w = TXS_W + 2 * TXS_DOCK_WALL + 0.8
        dock_outer_d = TXS_D + 2 * TXS_DOCK_WALL + 0.8
        # west/east rails
        for x in (TXS_X - dock_outer_w / 2 + TXS_DOCK_WALL / 2, TXS_X + dock_outer_w / 2 - TXS_DOCK_WALL / 2):
            with Locations((x, TXS_Y, FLOOR)):
                Box(TXS_DOCK_WALL, dock_outer_d, TXS_DOCK_H, mode=Mode.ADD, align=(Align.CENTER, Align.CENTER, Align.MIN))
        # south/north tiny end stops, with one gap implied by not making them full-height clips.
        for y in (TXS_Y - dock_outer_d / 2 + TXS_DOCK_WALL / 2, TXS_Y + dock_outer_d / 2 - TXS_DOCK_WALL / 2):
            with Locations((TXS_X, y, FLOOR)):
                Box(dock_outer_w, TXS_DOCK_WALL, TXS_DOCK_H * 0.75, mode=Mode.ADD, align=(Align.CENTER, Align.CENTER, Align.MIN))

        # Connector yoke pads: pads near D-sub side wings/flanges.
        # They should become real measured contact surfaces later.
        for x in (-YOKE_X, YOKE_X):
            with Locations((x, DSUB_Y, DSUB_Z + DSUB_H / 2 + 0.6)):
                Box(YOKE_PAD_W, YOKE_PAD_D, YOKE_PAD_H, mode=Mode.ADD, align=(Align.CENTER, Align.CENTER, Align.CENTER))

        # Diagonal-ish yoke ribs approximated as vertical web plates from side walls toward pads.
        # These make push-down load find the connector zone and pull-up load lift the connector wings.
        for sign in (-1, 1):
            x_mid = sign * (BOX_W / 2 - 22)
            x_pad = sign * YOKE_X
            # Use a hull-like rectangle path approximated with a long thin rib rotated in top view.
            dx = x_mid - x_pad
            dy = -10.0
            length = (dx * dx + dy * dy) ** 0.5
            angle = 0  # keep straight X rib for printability in this first model
            with Locations((sign * ((abs(x_mid) + abs(x_pad)) / 2), DSUB_Y - 7.0, FLOOR)):
                Box(length, YOKE_RIB_T, BOX_H - FLOOR - 2.0, mode=Mode.ADD, align=(Align.CENTER, Align.CENTER, Align.MIN))

        # Small snap sockets on the inside south wall for the lid lip tabs.
        for x in (-55, 55):
            with Locations((x, -BOX_D / 2 + WALL / 2, BOX_H - 5.0)):
                Box(14, WALL + 0.4, 2.2, mode=Mode.SUBTRACT, align=(Align.CENTER, Align.CENTER, Align.CENTER))

        safe_fillet(p.part.edges().filter_by(Axis.Z), 0.6)
        return p.part


def make_lid():
    """Snap lid with downward mounting posts for MPU and Pico."""
    with BuildPart() as p:
        # Flat top slab, slightly inset to pocket into bottom walls.
        Box(
            BOX_W - 2 * LID_CLEAR,
            BOX_D - 2 * LID_CLEAR,
            LID_THICK,
            align=(Align.CENTER, Align.CENTER, Align.MIN),
        )

        # Downward skirt that pockets inside bottom box.
        skirt_w = BOX_W - 2 * WALL - 2 * LID_CLEAR
        skirt_d = BOX_D - 2 * WALL - 2 * LID_CLEAR
        with Locations((0, 0, -LID_OVERLAP)):
            Box(skirt_w, skirt_d, LID_OVERLAP, mode=Mode.ADD, align=(Align.CENTER, Align.CENTER, Align.MIN))
        # Hollow out skirt center so it's a rim, not a plug.
        with Locations((0, 0, -LID_OVERLAP - 0.05)):
            Box(skirt_w - 2.0, skirt_d - 2.0, LID_OVERLAP + 0.1, mode=Mode.SUBTRACT, align=(Align.CENTER, Align.CENTER, Align.MIN))

        # North-center plinth/connector bite through lid too.
        with Locations((0, BOX_D / 2 - PLINTH_D / 2 + CLEAR, -LID_OVERLAP - 0.1)):
            Box(PLINTH_W + 2 * CLEAR, PLINTH_D + 2 * CLEAR, LID_THICK + LID_OVERLAP + 0.3, mode=Mode.SUBTRACT)

        # Front lip handle, kindergarten coloring-box style.
        with Locations((0, -BOX_D / 2 - LIP_D / 2 + 0.8, 0)):
            Box(LIP_W, LIP_D, LIP_H, mode=Mode.ADD, align=(Align.CENTER, Align.CENTER, Align.MIN))

        # Small snap tabs that fit bottom south-wall sockets.
        for x in (-55, 55):
            with Locations((x, -BOX_D / 2 + WALL / 2, -1.8)):
                Box(12, WALL, 1.8, mode=Mode.ADD, align=(Align.CENTER, Align.CENTER, Align.MIN))

        # MPU-6050 downward posts; pins presumed on west/left side, board hangs below lid.
        # Two holes are on east/right side, so posts sit just east of board centerline.
        for y in (MPU_Y - MPU_HOLE_SPAN_Y / 2, MPU_Y + MPU_HOLE_SPAN_Y / 2):
            with Locations((MPU_X + MPU_HOLE_X_FROM_CENTER, y, -MPU_POST_H)):
                Cylinder(MPU_POST_R, MPU_POST_H, mode=Mode.ADD, align=(Align.CENTER, Align.CENTER, Align.MIN))
        # LED light window above MPU.
        with Locations((MPU_X, MPU_Y, -0.1)):
            Cylinder(MPU_LED_WINDOW_R, LID_THICK + 0.2, mode=Mode.SUBTRACT, align=(Align.CENTER, Align.CENTER, Align.MIN))

        # Pico W hanging posts, generic four-corner-ish pads/pegs.
        for x in (PICO_X - PICO_W / 2 + 4, PICO_X + PICO_W / 2 - 4):
            for y in (PICO_Y - PICO_D / 2 + 4, PICO_Y + PICO_D / 2 - 4):
                with Locations((x, y, -PICO_POST_H)):
                    Cylinder(PICO_POST_R, PICO_POST_H, mode=Mode.ADD, align=(Align.CENTER, Align.CENTER, Align.MIN))

        # USB exit notch on north side of Pico, toward robot center / over connector.
        with Locations((PICO_X, BOX_D / 2 - USB_EXIT_D / 2, -0.1)):
            Box(USB_EXIT_W, USB_EXIT_D, LID_THICK + 1.0, mode=Mode.SUBTRACT, align=(Align.CENTER, Align.CENTER, Align.MIN))

        # Lid-side compression ribs above connector. Push anywhere, load routes toward D-sub zone.
        for sign in (-1, 1):
            with Locations((sign * 35, DSUB_Y - 5, -6.0)):
                Box(42, 2.0, 6.0, mode=Mode.ADD, align=(Align.CENTER, Align.CENTER, Align.MIN))

        safe_fillet(p.part.edges().filter_by(Axis.Z), 0.5)
        return p.part


bottom = make_bottom_box()
lid = make_lid()

# Place lid above bottom for inspection/export as an assembly-like compound.
with BuildPart() as assembly:
    add(bottom)
    with Locations((0, 0, BOX_H + 4)):
        add(lid)

# Export separate and combined files.
export_step(bottom, "create1_brainstem_row0_box_v4_bottom.step")
export_stl(bottom, "create1_brainstem_row0_box_v4_bottom.stl")
export_step(lid, "create1_brainstem_row0_box_v4_lid.step")
export_stl(lid, "create1_brainstem_row0_box_v4_lid.stl")
export_step(assembly.part, "create1_brainstem_row0_box_v4_assembly.step")
export_stl(assembly.part, "create1_brainstem_row0_box_v4_assembly.stl")

try:
    show_object(assembly.part)
except NameError:
    pass

print("Wrote:")
print("  create1_brainstem_row0_box_v4_bottom.step/.stl")
print("  create1_brainstem_row0_box_v4_lid.step/.stl")
print("  create1_brainstem_row0_box_v4_assembly.step/.stl")