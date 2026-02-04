import os
import numpy as np
import torch
from pathlib import Path
from PIL import Image as PILImage
from utils.general_utils import PILtoTorchMask
from utils.read_write_model import (
    read_points3D_binary, read_images_binary, read_cameras_binary, qvec2rotmat
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_mask(mask_path, resolution = None):
    """Load a mask image and return it as a binary numpy array."""
    mask = PILImage.open(mask_path)
    if resolution is not None:
        mask = PILtoTorchMask(mask, resolution)[0]
    
    if mask is None:
        raise FileNotFoundError(f"Could not load mask: {mask_path}")

    binary_mask = np.array(mask) > 127
    return binary_mask   

def get_track_ids(source_path):
    """
    Get the track IDs from the COLMAP sparse reconstruction directory.
    """
    points3D_path = os.path.join(source_path, "sparse/0/points3D.bin")
    points3D = read_points3D_binary(points3D_path)
    images_path = os.path.join(source_path, "sparse/0/images.bin")
    images = read_images_binary(images_path)

    num_images = len(images)+1
    track_ids = torch.zeros((len(points3D), num_images), dtype=torch.bool, device=device)

    for i, (point_id, point) in enumerate(points3D.items()):
        for img_id in point.image_ids:
            track_ids[i, img_id] = True

    return track_ids

def localize_gaussians(points, track_ids, source_path, visibility_threshold=1):
    """
    Get boolean mask of points which project onto text regions in the images at least `visibility_threshold` times.
    Args:
        points (torch.Tensor): Tensor of shape (N, 3) containing 3D points.
        track_ids (torch.Tensor): Tensor of shape (N, T) containing track IDs for each point.
        source_path (str): Path to the COLMAP sparse reconstruction directory.
        visibility_threshold (int): Minimum number of images a point must be visible in to be considered valid.
    Returns:
        torch.Tensor: Boolean tensor of shape (N,) where True indicates the point is visible in enough images.
    """
    
    visibility_counts = get_vis_counts(points, track_ids, source_path)
    return visibility_counts >= visibility_threshold

def get_vis_counts(points, track_ids, source_path):
    """
    Get visibility counts for each point based on mask projections.
    """
    
    masks_dir = os.path.join(source_path, "masks")
    if not os.path.exists(masks_dir):
        print(f"Mask directory does not exist: {masks_dir}")
        exit(1)
    
    cameras_path = os.path.join(source_path, "sparse/0/cameras.bin")
    cameras = read_cameras_binary(cameras_path)
    
    images_path = os.path.join(source_path, "sparse/0/images.bin")
    images = read_images_binary(images_path)
    
    points_tensor = points.clone().detach().to(device)
    
    visibility_counts = torch.zeros(len(points), dtype=torch.int32, device=device)

    mask_files = {}
    for img_id, img in images.items():
        mask_path = os.path.join(masks_dir, f"{Path(img.name)}")
        if os.path.exists(mask_path):
            mask_files[img_id] = mask_path

    
    for img_id, mask_path in mask_files.items():
        img = images[img_id]
        camera = cameras[img.camera_id]
        resolution = (camera.width, camera.height)
        mask = torch.from_numpy(load_mask(mask_path, resolution))
        if mask is None:
            continue
        
        if mask.sum() == 0:
            continue

        mask = mask.to(device)
        
        R = qvec2rotmat(img.qvec)
        R = torch.from_numpy(R).float().to(device)
        T = torch.from_numpy(img.tvec).float().to(device)

        visible_points_mask = track_ids[:, img_id] == True

        points_cam = torch.mm(points_tensor, R.T) + T
        z = points_cam[:, 2]
        valid_z = z > 0

        x_norm = points_cam[:, 0] / z
        y_norm = points_cam[:, 1] / z

        fx, fy, cx, cy = camera.params
        u = fx * x_norm + cx
        v = fy * y_norm + cy

        valid_bounds = (u >= 0) & (u < camera.width) & (v >= 0) & (v < camera.height)

        valid = valid_z & valid_bounds

        if valid.sum() == 0:
            del R, T, mask, points_cam, z, x_norm, y_norm, u, v, valid_z, valid_bounds, valid
            torch.cuda.empty_cache()
            continue

        u_int = u[valid].long()
        v_int = v[valid].long()

        visible_indices = torch.nonzero(valid, as_tuple=True)[0]

        mask_values = mask[v_int, u_int]
        mask_valid = (mask_values == 1)
        mask_valid = mask_valid & visible_points_mask[visible_indices]

        if mask_valid.sum() == 0:
            del R, T, mask, points_cam, z, x_norm, y_norm, u, v, valid_z, valid_bounds, valid, u_int, v_int, visible_indices, mask_values
            torch.cuda.empty_cache()
            continue

        visible_indices = visible_indices[mask_valid]
        visibility_counts[visible_indices] += 1

        del R, T, mask, points_cam, z, x_norm, y_norm, u, v, valid_z, valid_bounds, valid
        del u_int, v_int, visible_indices, mask_values, mask_valid
        torch.cuda.empty_cache()

    return visibility_counts
