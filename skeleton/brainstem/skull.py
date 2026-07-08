"""
skull.py

A build123d model for the Netherwick / Pete brainstem command cube.

Architecture:
- Low-profile cube cap centered on the Create 1 cargo-bay connector plinth.
- North wall has an exact-width plinth notch with a tiny lower lead-in/funnel.
- Pull backward/south to disengage; a small south shelf-cliff acts as a mild fulcrum.
- Bottom is a hollow open-top cube; top is a translucent-black hinged/snap lid.
- MPU-6050 and Pico W hang under the lid and remain visible through that lid.
- TXS0108E level shifter sits dead-bug on the floor, centered south of the plinth.
- No full-width row, no side pods, no screw dependency, no yoke drama.

This is a fit/design model, not a final measured part. The constants at top are meant
for fast iteration after one small print or FreeCAD inspection.
"""

from build123d import *
from pathlib import Path
import struct

# ------------------------------------------------------------
# Coordinate system
# X: west/east across row 0. Centered on connector/plinth.
# Y: south/north. South/front/open cargo bay is negative Y. North/hinge/robot center is positive Y.
# Z: up from shelf top.
# ------------------------------------------------------------

# Overall cube dimensions. Still deliberately conservative placeholders.
BOX_W = 88.0           # compact cap, not cargo-bay full width
BOX_D = 45.0           # enough south room for TXS dock and extraction grip
BOX_H = 16.0           # low-profile bottom box wall height
WALL = 2.2
FLOOR = 1.8
CLEAR = 0.45

# Plinth / D-sub north bite. The plinth sits along the north edge, centered in X.
PLINTH_W = 64.0        # exact nominal raised-plinth width before print clearance
PLINTH_D = 18.0        # how far south the bite intrudes from north wall
PLINTH_CLEAR_H = 12.0  # clearance height under/around connector shell
PLINTH_FUNNEL_EXTRA_W = 5.0
PLINTH_FUNNEL_D = 4.2
PLINTH_FUNNEL_H = 5.0

# D-sub male connector shell/flange approximation, used for clearance/yoke pads.
DSUB_W = 56.0
DSUB_D = 13.5
DSUB_H = 13.0
DSUB_Z = FLOOR + 4.5   # approximate shell center height inside box
DSUB_Y = BOX_D / 2 - PLINTH_D + 7.0

# South extraction shelf-cliff. Keep this shallow: it should pry gently, not lever
# the D-sub hard enough to bend pins.
FULCRUM_W = 24.0
FULCRUM_D = 5.0
FULCRUM_H = 2.6
FULCRUM_Y = -BOX_D / 2 - FULCRUM_D / 2 + 0.8

# Lid parameters
LID_THICK = 2.0
LID_OVERLAP = 1.2      # lid pockets down inside bottom box
LID_CLEAR = 0.25
HINGE_W = 10.0
HINGE_D = 3.4
HINGE_H = 3.0
SNAP_TAB_W = 10.0
SNAP_TAB_D = 2.0
SNAP_TAB_H = 1.7

# Board dimensions and hole positions.
#
# Pico source notes:
# - ncarandini/KiCad-RP-Pico gives the 21 x 51 mm board outline and SWD holes.
# - The official Raspberry Pi Pico mechanical drawing gives the four 2.1 mm
#   mounting holes. In normal Pico coordinates their centers are at
#   x = +/-5.7, y = +/-23.5. This enclosure keeps the Pico crosswise, so those
#   become local x = +/-23.5, y = +/-5.7.
#
# MPU/GY-521 source notes:
# - Common GY-521-style MPU-6050 footprint: board outline 16.51 x 20.574 mm,
#   two mounting holes on the side opposite the pin header, at local
#   x = +5.715 and y = +/-8.001 from board center.
MPU_W = 16.51
MPU_D = 20.574
MPU_HOLE_SPAN_Y = 16.002
MPU_HOLE_X_FROM_CENTER = 5.715
MPU_X = -29.0
MPU_Y = -6.0
MPU_POST_R = 1.25
MPU_POST_H = 6.0
MPU_LED_WINDOW_R = 2.0

PICO_W = 51.0
PICO_D = 21.0
PICO_X = 14.0
PICO_Y = -7.0
PICO_MOUNT_X = 23.5
PICO_MOUNT_Y = 5.7
PICO_MOUNT_HOLE_D = 2.1
PICO_POST_R = PICO_MOUNT_HOLE_D / 2 - 0.10
PICO_POST_H = 6.0
USB_EXIT_W = 14.0
USB_EXIT_D = 8.0
PICO_RESET_X = PICO_X - 18.0
PICO_RESET_Y = PICO_Y + 7.2
RESET_PUSHER_R = 2.4
RESET_PUSHER_POST_R = 1.0
RESET_PUSHER_POST_H = 4.0

# TXS0108E floor board. The pictured HW-221/CJMCU-style boards vary, but the
# closest documented breakout family is 26 x 15.5 mm with M2 mounting holes on
# a 22 mm pitch. The floor uses small locating pegs, not screw dependency.
TXS_W = 26.0
TXS_D = 15.5
TXS_X = 0.0
TXS_Y = -15.0
TXS_DOCK_WALL = 1.2
TXS_DOCK_H = 3.0
TXS_MOUNT_PITCH_X = 22.0
TXS_MOUNT_HOLE_D = 2.0
TXS_LOCATOR_PEG_R = TXS_MOUNT_HOLE_D / 2 - 0.10
TXS_LOCATOR_PEG_H = 1.6

# ASCII STL is larger than binary, but easier to inspect and friendlier to
# viewers/importers that get confused by OpenCASCADE binary STL headers.
STL_ASCII = True
STL_TOLERANCE = 0.001
STL_ANGULAR_TOLERANCE = 0.1


def safe_fillet(edges, radius):
    try:
        fillet(edges, radius=radius)
    except Exception as exc:
        print(f"Skipping fillet radius={radius}: {exc}")


def export_printable_stl(shape, path):
    ok = export_stl(
        shape,
        path,
        tolerance=STL_TOLERANCE,
        angular_tolerance=STL_ANGULAR_TOLERANCE,
        ascii_format=STL_ASCII,
    )
    if not ok:
        raise RuntimeError(f"STL export failed: {path}")
    return audit_stl(path)


def audit_stl(path):
    data = Path(path).read_bytes()
    if data.lstrip().startswith(b"solid"):
        triangles = data.count(b"facet normal")
        return f"{path}: ASCII STL, {triangles} facets, {len(data)} bytes"
    if len(data) < 84:
        raise RuntimeError(f"STL export is too small to contain a binary header: {path}")
    triangles = struct.unpack("<I", data[80:84])[0]
    return f"{path}: binary STL, {triangles} facets, {len(data)} bytes"


def make_bottom_box():
    """Bottom open-top command cube with north-center plinth bite."""
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

        # North-center exact-width plinth bite through the north wall and body.
        with Locations((0, BOX_D / 2 - PLINTH_D / 2 + CLEAR, 0)):
            Box(
                PLINTH_W + 2 * CLEAR,
                PLINTH_D + 2 * CLEAR,
                PLINTH_CLEAR_H,
                mode=Mode.SUBTRACT,
                align=(Align.CENTER, Align.CENTER, Align.MIN),
            )

        # Tiny lower lead-in/funnel at the south mouth of the notch. It is only
        # low on the wall, so the upper notch remains an exact-width locator.
        with Locations((0, BOX_D / 2 - PLINTH_D - PLINTH_FUNNEL_D / 2 + CLEAR, 0)):
            Box(
                PLINTH_W + PLINTH_FUNNEL_EXTRA_W + 2 * CLEAR,
                PLINTH_FUNNEL_D + 2 * CLEAR,
                PLINTH_FUNNEL_H,
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

        # Two low locator pegs for the TXS0108E mounting holes. These keep the
        # dead-bug board centered without making the enclosure depend on screws.
        for x in (TXS_X - TXS_MOUNT_PITCH_X / 2, TXS_X + TXS_MOUNT_PITCH_X / 2):
            with Locations((x, TXS_Y, FLOOR)):
                Cylinder(TXS_LOCATOR_PEG_R, TXS_LOCATOR_PEG_H, mode=Mode.ADD, align=(Align.CENTER, Align.CENTER, Align.MIN))

        # Small south shelf-cliff for pull-back extraction. It is narrow and low,
        # acting as a mild fulcrum while the D-sub lifts gently out of engagement.
        with Locations((0, FULCRUM_Y, 0)):
            Box(FULCRUM_W, FULCRUM_D, FULCRUM_H, mode=Mode.ADD, align=(Align.CENTER, Align.CENTER, Align.MIN))

        # Small snap sockets on the inside south wall for the lid tabs.
        for x in (-22, 22):
            with Locations((x, -BOX_D / 2 + WALL / 2, BOX_H - 5.0)):
                Box(SNAP_TAB_W + 2.0, WALL + 0.4, SNAP_TAB_H + 0.5, mode=Mode.SUBTRACT, align=(Align.CENTER, Align.CENTER, Align.CENTER))

        # North hinge receiver reliefs; the lid has matching knuckles.
        for x in (-22, 0, 22):
            with Locations((x, BOX_D / 2 - WALL / 2, BOX_H - 2.2)):
                Box(HINGE_W + 1.0, WALL + 0.8, 2.8, mode=Mode.SUBTRACT, align=(Align.CENTER, Align.CENTER, Align.CENTER))

        safe_fillet(p.part.edges().filter_by(Axis.Z), 0.25)
        return p.part


def make_lid():
    """Translucent-black style snap/hinge lid with downward board mounts."""
    with BuildPart() as p:
        # Flat top slab, slightly inset to pocket into bottom walls.
        Box(
            BOX_W - 2 * LID_CLEAR,
            BOX_D - 2 * LID_CLEAR,
            LID_THICK,
            align=(Align.CENTER, Align.CENTER, Align.MIN),
        )
        safe_fillet(p.part.edges().filter_by(Axis.Z), 0.4)

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

        # Small snap tabs that fit bottom south-wall sockets.
        for x in (-22, 22):
            with Locations((x, -BOX_D / 2 + WALL / 2, -SNAP_TAB_H)):
                Box(SNAP_TAB_W, SNAP_TAB_D, SNAP_TAB_H, mode=Mode.ADD, align=(Align.CENTER, Align.CENTER, Align.MIN))

        # Simple print-friendly hinge knuckles on the north edge.
        for x in (-22, 0, 22):
            with Locations((x, BOX_D / 2 + HINGE_D / 2 - 0.8, 0)):
                Box(HINGE_W, HINGE_D, HINGE_H, mode=Mode.ADD, align=(Align.CENTER, Align.CENTER, Align.MIN))

        # MPU-6050 downward posts; pins presumed on west/left side, board hangs below lid.
        # Two holes are on east/right side, so posts sit just east of board centerline.
        for y in (MPU_Y - MPU_HOLE_SPAN_Y / 2, MPU_Y + MPU_HOLE_SPAN_Y / 2):
            with Locations((MPU_X + MPU_HOLE_X_FROM_CENTER, y, -MPU_POST_H)):
                Cylinder(MPU_POST_R, MPU_POST_H, mode=Mode.ADD, align=(Align.CENTER, Align.CENTER, Align.MIN))
        # LED light window above MPU.
        with Locations((MPU_X, MPU_Y, -0.1)):
            Cylinder(MPU_LED_WINDOW_R, LID_THICK + 0.2, mode=Mode.SUBTRACT, align=(Align.CENTER, Align.CENTER, Align.MIN))

        # Pico W hanging posts at the real 2.1 mm mounting-hole centers.
        for x in (PICO_X - PICO_MOUNT_X, PICO_X + PICO_MOUNT_X):
            for y in (PICO_Y - PICO_MOUNT_Y, PICO_Y + PICO_MOUNT_Y):
                with Locations((x, y, -PICO_POST_H)):
                    Cylinder(PICO_POST_R, PICO_POST_H, mode=Mode.ADD, align=(Align.CENTER, Align.CENTER, Align.MIN))

        # USB exit notch on north side of Pico, toward robot center / over connector.
        with Locations((PICO_X, BOX_D / 2 - USB_EXIT_D / 2, -0.1)):
            Box(USB_EXIT_W, USB_EXIT_D, LID_THICK + 1.0, mode=Mode.SUBTRACT, align=(Align.CENTER, Align.CENTER, Align.MIN))

        # Reset pusher over the Pico reset position: small proud top button with
        # a short underside post that reaches toward the board switch.
        with Locations((PICO_RESET_X, PICO_RESET_Y, LID_THICK)):
            Cylinder(RESET_PUSHER_R, 0.9, mode=Mode.ADD, align=(Align.CENTER, Align.CENTER, Align.MIN))
        with Locations((PICO_RESET_X, PICO_RESET_Y, -RESET_PUSHER_POST_H)):
            Cylinder(RESET_PUSHER_POST_R, RESET_PUSHER_POST_H, mode=Mode.ADD, align=(Align.CENTER, Align.CENTER, Align.MIN))

        return p.part


bottom = make_bottom_box()
lid = make_lid()

# Place lid above bottom for inspection/export as an assembly-like compound.
with BuildPart() as assembly:
    add(bottom)
    with Locations((0, 0, BOX_H + 4)):
        add(lid)

# Export separate and combined files.
export_step(bottom, "skull_bottom.step")
bottom_stl = export_printable_stl(bottom, "skull_bottom.stl")
export_step(lid, "skull_lid.step")
lid_stl = export_printable_stl(lid, "skull_lid.stl")
export_step(assembly.part, "skull_assembly.step")
assembly_stl = export_printable_stl(assembly.part, "skull_assembly.stl")

try:
    show_object(assembly.part)
except NameError:
    pass

print("Wrote:")
print("  skull_bottom.step/.stl")
print("  skull_lid.step/.stl")
print("  skull_assembly.step/.stl")
print("STL audit:")
print(f"  {bottom_stl}")
print(f"  {lid_stl}")
print(f"  {assembly_stl}")
