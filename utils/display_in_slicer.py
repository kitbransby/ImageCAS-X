"""Paste-and-run script for the 3D Slicer Python console.

Two modes, set by MODE in the CONFIG block below:

  "segments"  one ImageCAS-X scan (multi-label mask + left/right centerlines),
              with each coronary segment (labels 1-14) coloured distinctly.

  "compare"   the qualitative figure: the same scan's binarised GT and each
              method's prediction, loaded on top of each other in the scan's own
              coordinates as one model per method, so they can be shown and
              hidden one at a time. Only the GT starts visible, and each node's
              name carries that method's Dice on the scan. Pick the scans worth
              showing by percentile of the summed Dice of the methods in
              COMPARE_RUNS — e.g. the 5th/50th/95th, read off the runs'
              <run>_results.json — then set SCAN_ID and BENCHMARK_DIR by hand.

Edit the CONFIG block, then paste the whole file into Slicer's Python console
(View > Python console).

Path conventions below match configs/pipeline.json: masks are
"<scan_id>.coronary.nii.gz", centerlines are
"<scan_id>.coronary_{left,right}_centerline.vtk", and predictions are
"<benchmark_dir>/<run>/predictions/<scan_id>.nii.gz" — the layout evaluate.py
scores.

Written against Slicer 5.8.1's Python API.
"""

import json
import os
import numpy as np
import vtk
import slicer

# ----------------------------- CONFIG ---------------------------------------
MODE = "segments"  # "segments" (GT coloured per coronary segment) or "compare"

DATA_ROOT = ""
SCAN_ID = ""  # edit to the scan you want to view

MASK_DIR = "segmentations"
MASK_SUFFIX = ".coronary.nii.gz"
CENTERLINES_DIR = "centerlines"

# --- compare mode -----------------------------------------------------------
# Folder holding one sub-folder per run, each with predictions/<scan_id>.nii.gz
# and <run>_results.json.
BENCHMARK_DIR = ""
PREDICTION_DIR = "predictions"
PREDICTION_SUFFIX = ".nii.gz"
# (run folder, display name), drawn left to right after the ground truth. These
# must be the same methods the scan was selected on, or the figure will not show
# what the percentile was computed over.
COMPARE_RUNS = (
    ("cas_net", "CAS-Net"),
    ("ade_htl", "ADE-HTL"),
    ("nnunet", "nnU-Net"),
)
GT_LABEL = "Ground truth"
# The reference tree is neutral; the three predictions take the categorical
# colours the manuscript's figures use for methods (slots 1-3 of a
# CVD-validated palette).
GT_COLOR_HEX = "#8f8d87"
COMPARE_COLORS_HEX = {"cas_net": "#2a78d6", "ade_htl": "#eb6834", "nnunet": "#1baf7a"}
COMPARE_FALLBACK_HEX = "#8f8d87"

CENTERLINE_COLOR = (0.0, 0.0, 0.0)
CENTERLINE_LINE_WIDTH = 1.5

BACKGROUND_COLOR = (1.0, 1.0, 1.0)  # 3D view background, top and bottom

# Exported segment-model display settings (Slicer defaults, then these overrides).
MODEL_OPACITY = 0.5
MODEL_AMBIENT = 0.2
MODEL_DIFFUSE = 0.85
MODEL_SPECULAR = 0.6
MODEL_POWER = 12
MODEL_METALLIC = 0
MODEL_ROUGHNESS = 0

# Distinct colour per coronary segment label 1-14.
SEGMENT_COLORS_HEX = {
    1: "#cecece", 2: "#5a4ff1", 3: "#0fb114", 4: "#007afd", 5: "#00ffee",
    6: "#00631c", 7: "#afee00", 8: "#797979", 9: "#d83d29", 10: "#f5e944",
    11: "#f8ac13", 12: "#91ccd8", 13: "#b995d8", 14: "#4a4a4a",
}


def _hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


SEGMENT_COLORS = {label: _hex_to_rgb(hex_color) for label, hex_color in SEGMENT_COLORS_HEX.items()}
# ------------------------------------------------------------------------- -


def _path(directory, suffix):
    return os.path.join(DATA_ROOT, directory, f"{SCAN_ID}{suffix}")


def load_segmentation():
    mask_path = _path(MASK_DIR, MASK_SUFFIX)
    if not os.path.exists(mask_path):
        print(f"[error] mask not found: {mask_path}")
        return None

    labelmapNode = slicer.util.loadLabelVolume(mask_path)
    labelArray = slicer.util.arrayFromVolume(labelmapNode)

    segmentationNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
    segmentationNode.SetName(f"{SCAN_ID}_segmentation")
    segmentationNode.CreateDefaultDisplayNodes()
    segmentationNode.SetReferenceImageGeometryParameterFromVolumeNode(labelmapNode)
    segmentation = segmentationNode.GetSegmentation()

    for label_value in sorted(int(v) for v in np.unique(labelArray) if v != 0):
        color = SEGMENT_COLORS.get(label_value, (0.8, 0.8, 0.8))
        name = f"segment_{label_value}"
        segment_id = segmentation.AddEmptySegment(name, name, color)
        binary_array = (labelArray == label_value).astype(np.uint8)
        slicer.util.updateSegmentBinaryLabelmapFromArray(binary_array, segmentationNode, segment_id, labelmapNode)

    segmentationNode.CreateClosedSurfaceRepresentation()
    displayNode = segmentationNode.GetDisplayNode()
    displayNode.SetPreferredDisplayRepresentationName3D("Closed surface")
    displayNode.SetAllSegmentsVisibility(True)

    slicer.mrmlScene.RemoveNode(labelmapNode)
    return segmentationNode


def load_centerline(side):
    path = os.path.join(DATA_ROOT, CENTERLINES_DIR, f"{SCAN_ID}.coronary_{side}_centerline.vtk")
    if not os.path.exists(path):
        print(f"[warn] {side} centerline not found: {path}")
        return None

    modelNode = slicer.util.loadModel(path)
    modelNode.SetName(f"{SCAN_ID}_{side}_centerline")

    displayNode = modelNode.GetDisplayNode()
    displayNode.SetScalarVisibility(False)
    displayNode.SetColor(*CENTERLINE_COLOR)
    displayNode.SetLineWidth(CENTERLINE_LINE_WIDTH)
    displayNode.SetOpacity(1.0)
    displayNode.SetVisibility(True)
    return modelNode


def export_segments_to_models(segmentationNode):
    if segmentationNode is None:
        return

    visibleSegmentIds = vtk.vtkStringArray()
    segmentationNode.GetDisplayNode().GetVisibleSegmentIDs(visibleSegmentIds)

    shNode = slicer.mrmlScene.GetSubjectHierarchyNode()
    exportFolderItemId = shNode.CreateFolderItem(shNode.GetSceneItemID(), f"{SCAN_ID}_models")
    slicer.modules.segmentations.logic().ExportSegmentsToModels(segmentationNode, visibleSegmentIds, exportFolderItemId)

    childItemIds = vtk.vtkIdList()
    shNode.GetItemChildren(exportFolderItemId, childItemIds)
    for i in range(childItemIds.GetNumberOfIds()):
        modelNode = shNode.GetItemDataNode(childItemIds.GetId(i))
        if not isinstance(modelNode, slicer.vtkMRMLModelNode):
            continue
        style_model_display(modelNode.GetDisplayNode())

    # the exported models now carry the segmentation's appearance, so hide the
    # original surface to stop it overlapping them
    segmentationNode.GetDisplayNode().SetVisibility(False)


def style_model_display(displayNode):
    displayNode.SetOpacity(MODEL_OPACITY)
    displayNode.SetAmbient(MODEL_AMBIENT)
    displayNode.SetDiffuse(MODEL_DIFFUSE)
    displayNode.SetSpecular(MODEL_SPECULAR)
    displayNode.SetPower(MODEL_POWER)
    displayNode.SetMetallic(MODEL_METALLIC)
    displayNode.SetRoughness(MODEL_ROUGHNESS)


def setup_3d_view(anterior=False):
    layoutManager = slicer.app.layoutManager()
    layoutManager.setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutOneUp3DView)
    threeDWidget = layoutManager.threeDWidget(0)
    viewNode = threeDWidget.mrmlViewNode()
    viewNode.SetBackgroundColor(*BACKGROUND_COLOR)
    viewNode.SetBackgroundColor2(*BACKGROUND_COLOR)
    viewNode.SetBoxVisible(False)
    viewNode.SetAxisLabelsVisible(False)
    if anterior:
        # A fixed starting viewpoint, so the same scan frames the same way each
        # time rather than however the view was last rotated.
        import ctk
        threeDWidget.threeDView().lookFromViewAxis(ctk.ctkAxesWidget.Anterior)
    threeDWidget.threeDView().resetFocalPoint()
    threeDWidget.threeDView().resetCamera()


# ---------------------------- compare mode ----------------------------------

def load_binary_surface(mask_path, name, color):
    """One binary tree as a model node: mask -> single segment -> closed surface
    -> exported model, so it carries the same appearance as the segment models
    the other mode produces.
    """
    if not os.path.exists(mask_path):
        print(f"[warn] mask not found: {mask_path}")
        return None

    labelmapNode = slicer.util.loadLabelVolume(mask_path)
    labelArray = slicer.util.arrayFromVolume(labelmapNode)
    binary = (labelArray != 0)
    if not binary.any():
        print(f"[warn] {name}: mask is empty after binarising — nothing to show")
        slicer.mrmlScene.RemoveNode(labelmapNode)
        return None

    segmentationNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
    segmentationNode.SetName(f"{name}_segmentation")
    segmentationNode.CreateDefaultDisplayNodes()
    segmentationNode.SetReferenceImageGeometryParameterFromVolumeNode(labelmapNode)
    segment_id = segmentationNode.GetSegmentation().AddEmptySegment(name, name, color)
    slicer.util.updateSegmentBinaryLabelmapFromArray(
        binary.astype(np.uint8), segmentationNode, segment_id, labelmapNode)
    segmentationNode.CreateClosedSurfaceRepresentation()
    slicer.mrmlScene.RemoveNode(labelmapNode)

    # A copy, not the segmentation's own polydata: the segmentation node is
    # removed below and would take its representation with it.
    surface = vtk.vtkPolyData()
    surface.DeepCopy(segmentationNode.GetClosedSurfaceInternalRepresentation(segment_id))
    modelNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", name)
    modelNode.SetAndObservePolyData(surface)
    modelNode.CreateDefaultDisplayNodes()
    displayNode = modelNode.GetDisplayNode()
    displayNode.SetColor(*color)
    style_model_display(displayNode)
    # The segmentation was only a route to the surface; leaving it visible would
    # draw a second copy of the tree in the same place.
    slicer.mrmlScene.RemoveNode(segmentationNode)
    return modelNode


def run_dice(run_folder):
    """That run's Dice on SCAN_ID, x100, or None if it has no score for the scan.

    Read from <BENCHMARK_DIR>/<run>/<run>_results.json, the file evaluate.py
    writes, so the number in a node's name is the number in the stats table.
    """
    path = os.path.join(BENCHMARK_DIR, run_folder, f"{run_folder}_results.json")
    if not os.path.exists(path):
        print(f"[warn] no results file at {path} — node will be named without Dice")
        return None
    with open(path, encoding="utf-8") as f:
        per_scan = json.load(f).get("per_scan", {})
    metrics = per_scan.get(str(SCAN_ID)) or per_scan.get(SCAN_ID)
    if not metrics or metrics.get("dice") is None:
        print(f"[warn] {run_folder} has no Dice for scan {SCAN_ID}")
        return None
    return float(metrics["dice"]) * 100


def compare_predictions():
    """Binarised GT and every run's prediction for SCAN_ID, loaded in place.

    Nothing is moved or annotated: the four trees sit on top of each other in
    the scan's own coordinates, so toggling one model's visibility against
    another's in the Models module (or the subject hierarchy) shows exactly
    where they differ. Each node's name carries its Dice, which is what the
    visibility list is keyed by while switching between them.
    """
    if not BENCHMARK_DIR:
        print("[error] set BENCHMARK_DIR to the folder holding one sub-folder per run")
        return

    specs = [("gt", GT_LABEL, _path(MASK_DIR, MASK_SUFFIX), None)]
    for folder, label in COMPARE_RUNS:
        specs.append((folder, label,
                      os.path.join(BENCHMARK_DIR, folder, PREDICTION_DIR,
                                   f"{SCAN_ID}{PREDICTION_SUFFIX}"),
                      run_dice(folder)))

    loaded = []
    for key, label, path, dice in specs:
        color = _hex_to_rgb(GT_COLOR_HEX if key == "gt"
                            else COMPARE_COLORS_HEX.get(key, COMPARE_FALLBACK_HEX))
        name = label if dice is None else f"{label} (DSC {dice:.1f})"
        modelNode = load_binary_surface(path, name, color)
        if modelNode is None:
            continue
        # Only the ground truth starts visible: four overlapping trees drawn at
        # once are unreadable, and one visible tree is the state to toggle from.
        modelNode.GetDisplayNode().SetVisibility(key == "gt")
        loaded.append(name)

    if not loaded:
        print("[error] nothing loaded — check DATA_ROOT, BENCHMARK_DIR and SCAN_ID")
        return

    setup_3d_view(anterior=True)
    print(f"[done] scan {SCAN_ID}, showing {loaded[0]}; toggle visibility in the "
          f"Models module: " + ",  ".join(loaded))


def main():
    if MODE == "compare":
        compare_predictions()
        return
    segmentationNode = load_segmentation()
    load_centerline("left")
    load_centerline("right")
    setup_3d_view()
    export_segments_to_models(segmentationNode)
    print(f"[done] loaded scan {SCAN_ID}")


main()
