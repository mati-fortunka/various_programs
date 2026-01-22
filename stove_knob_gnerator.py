# -*- coding: utf-8 -*-

# Macro to generate a parametric stove knob in FreeCAD

import FreeCAD
import Part
from FreeCAD import Base

# --- Module metadata ---
__Name__ = "StoveKnob"
__Comment__ = "Creates a parametric stove knob with a D-shaped hole and fillets."
__Author__ = "Gemini"
__Version__ = "1.8"
__Date__ = "2023-10-27"
__License__ = "LGPL-2.0-or-later"
__Web__ = ""
__Wiki__ = ""
__Icon__ = ""
__Help__ = "Run this macro to create a stove knob object in the active document."
__Status__ = "Stable"
__Requires__ = "FreeCAD >= 0.19"
__Communication__ = "https://github.com/FreeCAD/FreeCAD-macros"
__Files__ = ""


# --- Class definition for the parametric knob object ---

class StoveKnob:
    """
    This class defines the parametric stove knob feature.
    It is instantiated for each StoveKnob object you create.
    """

    def __init__(self, obj):
        """
        The constructor is called when a new object is created.
        It defines the object's properties.
        """
        self.Type = "StoveKnob"
        obj.Proxy = self

        # Add properties to the FreeCAD object. These will be editable in the Property Editor.
        # --- Base Properties ---
        obj.addProperty("App::PropertyLength", "BaseRadius", "Knob Base",
                        "Radius of the circular base.").BaseRadius = 25.0
        obj.addProperty("App::PropertyLength", "BaseHeight", "Knob Base",
                        "Thickness of the circular base.").BaseHeight = 5.0

        # --- Handle Properties ---
        obj.addProperty("App::PropertyLength", "HandleLength", "Handle",
                        "Length of the top handle (along X-axis).").HandleLength = 40.0
        obj.addProperty("App::PropertyLength", "HandleWidth", "Handle",
                        "Width of the top handle (along Y-axis).").HandleWidth = 9.0
        obj.addProperty("App::PropertyLength", "HandleHeight", "Handle",
                        "Height of the top handle.").HandleHeight = 15.0

        # --- Hole Properties ---
        obj.addProperty("App::PropertyLength", "HoleRadius", "Hole",
                        "Radius of the circular part of the hole.").HoleRadius = 4.0
        obj.addProperty("App::PropertyLength", "HoleFlatOffset", "Hole",
                        "Distance from hole center to its flat side.").HoleFlatOffset = 3.0
        obj.addProperty("App::PropertyLength", "HoleDepth", "Hole",
                        "Depth of the hole from the bottom.").HoleDepth = 15.0
        obj.addProperty("App::PropertyLength", "HoleChamfer", "Hole",
                        "Size of the 45-degree chamfer at the hole opening.").HoleChamfer = 0.5
        obj.addProperty("App::PropertyBool", "ShowHole", "Hole", "Show the hole cutout.").ShowHole = True

        # --- Smoothing Properties ---
        obj.addProperty("App::PropertyLength", "HandleFilletRadius", "Smoothing",
                        "Radius for smoothing the handle edges.").HandleFilletRadius = 1.0
        obj.addProperty("App::PropertyLength", "BaseFilletRadius", "Smoothing",
                        "Radius for smoothing the junction of handle and base.").BaseFilletRadius = 2.0

    def execute(self, fp):
        """
        This method is called whenever the object needs to be recomputed.
        'fp' is the feature python object itself.
        """
        FreeCAD.Console.PrintMessage("Recomputing StoveKnob...\n")

        # --- Step 1: Create the circular base ---
        base = Part.makeCylinder(fp.BaseRadius, fp.BaseHeight)

        # --- Step 2: Create the handle and apply fillets to it BEFORE fusion ---
        handle = Part.makeBox(fp.HandleLength, fp.HandleWidth, fp.HandleHeight)

        handle_fillet_val = fp.HandleFilletRadius.Value
        if handle_fillet_val > 0.01:
            edges_to_fillet_handle = []
            for edge in handle.Edges:
                # Find top edges and vertical edges of the simple box
                is_top_edge = abs(edge.BoundBox.ZMin - fp.HandleHeight.Value) < 0.01 and abs(
                    edge.BoundBox.ZMax - fp.HandleHeight.Value) < 0.01

                v_start = edge.Vertexes[0].Point
                v_end = edge.Vertexes[1].Point
                is_vertical = abs(v_start.x - v_end.x) < 0.01 and abs(v_start.y - v_end.y) < 0.01

                if is_top_edge or is_vertical:
                    edges_to_fillet_handle.append(edge)

            if edges_to_fillet_handle:
                try:
                    handle = handle.makeFillet(handle_fillet_val, edges_to_fillet_handle)
                except Part.OCCError as e:
                    FreeCAD.Console.PrintWarning(f"Could not apply handle fillet: {e}\n")

        # --- Step 3: Position the (now possibly filleted) handle asymmetrically ---
        handle_x_pos = -fp.BaseRadius.Value
        handle_y_pos = -fp.HandleWidth.Value / 2
        handle_z_pos = fp.BaseHeight.Value
        handle.translate(Base.Vector(handle_x_pos, handle_y_pos, handle_z_pos))

        # --- Step 4: Fuse the base and handle ---
        knob_body = base.fuse(handle)
        last_valid_body = knob_body  # Backup in case the next fillet fails

        # --- Step 5: Apply the fillet where the handle meets the base ---
        base_fillet_val = fp.BaseFilletRadius.Value
        if base_fillet_val > 0.01:
            if not knob_body.Solids:
                FreeCAD.Console.PrintWarning("Fusion resulted in no solids. Aborting base fillet.\n")
            else:
                solid_body = knob_body.Solids[0]
                try:
                    section_plane = Part.Plane(Base.Vector(0, 0, fp.BaseHeight.Value), Base.Vector(0, 0, 1))
                    section_face = Part.Face(section_plane)
                    section = solid_body.section(section_face)

                    edges_to_fillet_base = []
                    for section_edge in section.Edges:
                        is_outer_circle = isinstance(section_edge.Curve, Part.Circle) and abs(
                            section_edge.Curve.Radius - fp.BaseRadius.Value) < 0.01
                        if not is_outer_circle:
                            # Find the corresponding 3D edge using the isSame method
                            for solid_edge in solid_body.Edges:
                                if section_edge.isSame(solid_edge):
                                    edges_to_fillet_base.append(solid_edge)
                                    break

                    if edges_to_fillet_base:
                        knob_body = solid_body.makeFillet(base_fillet_val, edges_to_fillet_base)
                        if knob_body.isNull() or not knob_body.isValid():
                            raise Part.OCCError("Base fillet resulted in invalid shape")
                        last_valid_body = knob_body
                    else:
                        FreeCAD.Console.PrintWarning("Could not find junction edges for base fillet.\n")
                except Exception as e:
                    FreeCAD.Console.PrintWarning(f"Could not apply base fillet: {e}\n")
                    knob_body = last_valid_body

        # --- Step 6: Create the D-shaped hole (now a blind hole) ---
        if fp.ShowHole:
            hole_cylinder = Part.makeCylinder(fp.HoleRadius, fp.HoleDepth)
            cutter_size = fp.HoleRadius.Value * 2.5
            cutter_box = Part.makeBox(cutter_size, cutter_size, fp.HoleDepth)
            cutter_box.translate(Base.Vector(-cutter_size / 2, fp.HoleFlatOffset.Value, 0))
            d_shape_hole = hole_cylinder.cut(cutter_box)

            final_shape = knob_body.cut(d_shape_hole)

            chamfer_val = fp.HoleChamfer.Value
            if chamfer_val > 0.01:
                hole_edges = [e for e in final_shape.Edges if
                              abs(e.BoundBox.ZMax - 0) < 0.01 and abs(e.BoundBox.ZMin - 0) < 0.01]
                if hole_edges:
                    try:
                        final_shape = final_shape.makeChamfer(chamfer_val, hole_edges)
                    except Part.OCCError:
                        FreeCAD.Console.PrintWarning("Could not apply hole chamfer. Radius might be too large.\n")
        else:
            final_shape = knob_body

        # --- Step 9: Assign the final shape to the object ---
        fp.Shape = final_shape

    def onChanged(self, fp, prop):
        """
        This method is called when a property of the object is changed.
        """
        if prop in ["BaseRadius", "BaseHeight", "HandleLength", "HandleWidth",
                    "HandleHeight", "HoleRadius", "HoleFlatOffset", "ShowHole",
                    "HoleDepth", "HoleChamfer", "HandleFilletRadius", "BaseFilletRadius"]:
            fp.touch()


def makeStoveKnob():
    """
    Helper function to create a new StoveKnob object in the active document.
    """
    doc = FreeCAD.activeDocument()
    if doc is None:
        doc = FreeCAD.newDocument("StoveKnobDoc")

    obj = doc.addObject("Part::FeaturePython", "StoveKnob")
    StoveKnob(obj)
    obj.ViewObject.Proxy = 0
    doc.recompute()
    return obj


# --- Main execution block ---
if __name__ == "__main__":
    makeStoveKnob()