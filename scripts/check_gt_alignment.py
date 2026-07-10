# check_gt_alignment.py
#
# Standalone GT-alignment check: draws the recorded ground-truth silhouette
# (rendered from the mesh at cam_R_m2c/cam_t_m2c) plus a pose-axis gizmo on
# top of the first N raw RGB frames, for camera_l/camera_0/camera_1, given a
# BOP-format dataset directory. Saves one overlay image per camera/frame
# plus a single combined grid image for quick side-by-side review.
#
# Exists to let you manually re-test alignment (e.g. while tuning
# ApproximateTimeSynchronizer's `slop` in RosClients.py) without needing a
# round-trip through the diff-rendering side of this project -- only numpy
# and OpenCV are required, both of which this ROS2 environment already has
# via cv_bridge.
#
# Dataset layout expected (standard BOP, same as BopSampleSaver produces):
#   <dataset_root>/<camera_name>/000001/{rgb,scene_camera.json,scene_gt.json}
#
# Usage:
#   python scripts/check_gt_alignment.py --dataset-root /path/to/dataset
#   python scripts/check_gt_alignment.py --dataset-root /path/to/dataset \
#       --mesh-path "/path/to/tool pitch link.OBJ" --frames 2 --obj-ids 1,3
#
# Output: written to <dataset_root>/gt_alignment_check/ -- one PNG per
# camera/frame/object plus grid_overview.png (all cameras x all frames,
# first matched object per cell). Also attempts to open a window via
# cv2.imshow; if there's no display attached (e.g. plain SSH) that's caught
# and skipped, saved files are still there either way.

import argparse
import json
import os

import cv2
import numpy as np


DEFAULT_CAMERAS = ["camera_l", "camera_0", "camera_1"]
DEFAULT_MESH_PATH = (
    "../../surgical_robotics_challenge/ADF/PSMs/LND_420006/high_res/tool pitch link.OBJ"
)


# ─────────────────────────── mesh loading (no pytorch3d) ──────────────────────
def load_obj_simple(path: str):
    """Minimal OBJ parser: vertices + triangulated faces (fan triangulation
    for any n-gon), ignoring vt/vn/material references entirely."""
    verts = []
    faces = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("v "):
                parts = line.split()
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("f "):
                idx = [int(p.split("/")[0]) - 1 for p in line.split()[1:]]
                for i in range(1, len(idx) - 1):
                    faces.append([idx[0], idx[i], idx[i + 1]])
    verts = np.array(verts, dtype=np.float64)
    faces = np.array(faces, dtype=np.int64)
    if np.abs(verts).max() < 1.0:
        verts = verts * 1000.0  # m -> mm, matching the diff-rendering scripts' convention
    return verts, faces


# ─────────────────────────── silhouette rendering ─────────────────────────────
def render_silhouette(verts, faces, R, t, cam_K, img_w, img_h):
    """Project verts with R,t (cam_R_m2c/cam_t_m2c, OpenCV convention) and
    rasterize each triangle into a binary mask. No backface culling / depth
    test needed for a silhouette -- overlapping faces just paint the same
    foreground pixel twice."""
    v_cam = (R @ verts.T).T + t  # (N,3), mm
    z = v_cam[:, 2]
    valid_v = z > 1e-3
    fx, cx, fy, cy = cam_K[0], cam_K[2], cam_K[4], cam_K[5]
    safe_z = np.clip(z, 1e-3, None)
    u = fx * v_cam[:, 0] / safe_z + cx
    v = fy * v_cam[:, 1] / safe_z + cy
    pts2d = np.stack([u, v], axis=1)

    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    face_valid = valid_v[faces].all(axis=1)
    for tri in faces[face_valid]:
        pts = pts2d[tri].astype(np.int32)
        cv2.fillConvexPoly(mask, pts, 255)
    return mask


def project_point(R, t, cam_K, p_local):
    p_cam = R @ np.array(p_local) + t
    if p_cam[2] <= 1e-3:
        return None
    fx, cx, fy, cy = cam_K[0], cam_K[2], cam_K[4], cam_K[5]
    return (fx * p_cam[0] / p_cam[2] + cx, fy * p_cam[1] / p_cam[2] + cy)


def draw_axis_gizmo(img_bgr, R, t, cam_K, length_mm=5.0):
    origin = project_point(R, t, cam_K, [0, 0, 0])
    if origin is None:
        return img_bgr
    axes = [
        ([length_mm, 0, 0], (0, 0, 255)),   # X red
        ([0, length_mm, 0], (0, 255, 0)),   # Y green
        ([0, 0, length_mm], (255, 0, 0)),   # Z blue
    ]
    ox, oy = int(round(origin[0])), int(round(origin[1]))
    for p_local, color in axes:
        p = project_point(R, t, cam_K, p_local)
        if p is None:
            continue
        px, py = int(round(p[0])), int(round(p[1]))
        cv2.line(img_bgr, (ox, oy), (px, py), color, 2)
    return img_bgr


# ─────────────────────────── main ──────────────────────────────────────────────
def process_camera(dataset_root, cam_name, num_frames, obj_ids, mesh_cache, out_dir):
    cam_dir = os.path.join(dataset_root, cam_name, "000001")
    scene_cam_path = os.path.join(cam_dir, "scene_camera.json")
    scene_gt_path = os.path.join(cam_dir, "scene_gt.json")
    rgb_dir = os.path.join(cam_dir, "rgb")

    if not (os.path.isfile(scene_cam_path) and os.path.isfile(scene_gt_path)):
        print(f"[skip] {cam_name}: missing scene_camera.json/scene_gt.json under {cam_dir}")
        return {}

    with open(scene_cam_path) as f:
        scene_cam = json.load(f)
    with open(scene_gt_path) as f:
        scene_gt = json.load(f)

    frame_keys = sorted(scene_gt.keys(), key=int)[:num_frames]
    results = {}
    for fkey in frame_keys:
        rgb_path = os.path.join(rgb_dir, f"{int(fkey):06d}.png")
        if not os.path.isfile(rgb_path):
            print(f"[skip] {cam_name} frame {fkey}: no rgb file at {rgb_path}")
            continue
        img = cv2.imread(rgb_path)
        if img is None:
            print(f"[skip] {cam_name} frame {fkey}: failed to read {rgb_path}")
            continue
        img_h, img_w = img.shape[:2]
        cam_K = scene_cam[fkey]["cam_K"]

        objs = [o for o in scene_gt[fkey] if o["obj_id"] in obj_ids]
        if not objs:
            print(f"[skip] {cam_name} frame {fkey}: none of obj_ids {obj_ids} present")
            continue

        overlay = img.copy()
        for obj in objs:
            R_gt = np.array(obj["cam_R_m2c"], dtype=np.float64).reshape(3, 3)
            t_gt = np.array(obj["cam_t_m2c"], dtype=np.float64)
            verts, faces = mesh_cache["verts"], mesh_cache["faces"]

            mask = render_silhouette(verts, faces, R_gt, t_gt, cam_K, img_w, img_h)
            contour_mask = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
            overlay[contour_mask > 0] = (255, 0, 255)  # magenta contour, BGR
            overlay = draw_axis_gizmo(overlay, R_gt, t_gt, cam_K)

        label = f"{cam_name} frame={fkey} obj_ids={[o['obj_id'] for o in objs]}"
        cv2.putText(overlay, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

        out_path = os.path.join(out_dir, f"gt_check_{cam_name}_f{fkey}.png")
        cv2.imwrite(out_path, overlay)
        print(f"[ok] {cam_name} frame {fkey}: saved {out_path}")
        results[fkey] = overlay
    return results


def build_grid(all_results, cameras, out_dir):
    """Tile every saved overlay into one grid image: rows=cameras, cols=frames."""
    frame_keys = sorted({fk for cam_res in all_results.values() for fk in cam_res.keys()}, key=int)
    if not frame_keys:
        return
    cell_w, cell_h = 320, 240
    grid = np.zeros((cell_h * len(cameras), cell_w * len(frame_keys), 3), dtype=np.uint8)
    for row, cam_name in enumerate(cameras):
        for col, fkey in enumerate(frame_keys):
            img = all_results.get(cam_name, {}).get(fkey)
            if img is None:
                continue
            resized = cv2.resize(img, (cell_w, cell_h))
            grid[row * cell_h:(row + 1) * cell_h, col * cell_w:(col + 1) * cell_w] = resized
    out_path = os.path.join(out_dir, "grid_overview.png")
    cv2.imwrite(out_path, grid)
    print(f"[ok] combined grid saved: {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, help="Path to the BOP dataset root")
    parser.add_argument("--mesh-path", default=DEFAULT_MESH_PATH,
                        help="Path to the tool pitch link OBJ (default: relative guess, edit if wrong)")
    parser.add_argument("--frames", type=int, default=2, help="Number of leading frames to check (default: 2)")
    parser.add_argument("--obj-ids", default="1,3", help="Comma-separated obj_ids to render (default: 1,3)")
    parser.add_argument("--cameras", default=",".join(DEFAULT_CAMERAS),
                        help="Comma-separated camera folder names (default: camera_l,camera_0,camera_1)")
    parser.add_argument("--no-display", action="store_true", help="Skip attempting cv2.imshow")
    args = parser.parse_args()

    obj_ids = {int(x) for x in args.obj_ids.split(",")}
    cameras = [c.strip() for c in args.cameras.split(",")]

    if not os.path.isfile(args.mesh_path):
        raise SystemExit(f"Mesh not found at '{args.mesh_path}' -- pass the correct path with --mesh-path")

    verts, faces = load_obj_simple(args.mesh_path)
    print(f"Mesh loaded: {len(verts)} verts, {len(faces)} faces")
    mesh_cache = {"verts": verts, "faces": faces}

    out_dir = os.path.join(args.dataset_root, "gt_alignment_check")
    os.makedirs(out_dir, exist_ok=True)

    all_results = {}
    for cam_name in cameras:
        all_results[cam_name] = process_camera(
            args.dataset_root, cam_name, args.frames, obj_ids, mesh_cache, out_dir
        )

    grid_path = build_grid(all_results, cameras, out_dir)

    if not args.no_display and grid_path:
        try:
            grid_img = cv2.imread(grid_path)
            cv2.imshow("GT alignment check (rows=cameras, cols=frames)", grid_img)
            print("Press any key in the image window to close it...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except cv2.error as e:
            print(f"[info] cv2.imshow unavailable (no display attached?): {e}")
            print(f"       Open the saved files directly instead, e.g.: {grid_path}")


if __name__ == "__main__":
    main()
