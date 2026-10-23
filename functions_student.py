"""Image registration utilities for the brain-image-registration notebook.

Reconstructed from project1.ipynb and its original functions_student.py template.
The missing implementations are NEW reconstructions, not a recovery of unsaved code.
Points are interpreted as (x, y), except that RANSAC works consistently
with either (x, y) or (row, col) if both sets use the same convention.
"""
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt


def get_affine_transformation(points_in, points_out):
    """Return a 3x3 affine matrix mapping (x,y) points_in to points_out."""
    src = np.asarray(points_in, dtype=np.float64)
    dst = np.asarray(points_out, dtype=np.float64)
    if src.ndim != 2 or src.shape[1] != 2 or src.shape != dst.shape or len(src) < 3:
        raise ValueError('Expected matching Nx2 point arrays with N >= 3')
    design = np.column_stack((src, np.ones(len(src))))
    coeff, _, rank, _ = np.linalg.lstsq(design, dst, rcond=None)
    if rank < 3:
        raise ValueError('Points do not determine an affine transformation')
    matrix = np.eye(3)
    matrix[:2, :] = coeff.T
    return matrix


def transform_points(points, matrix):
    pts = np.asarray(points, dtype=float)
    matrix = np.asarray(matrix, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2 or matrix.shape != (3, 3):
        raise ValueError('Expected Nx2 points and a 3x3 matrix')
    homogeneous = np.column_stack((pts, np.ones(len(pts))))
    result = homogeneous @ matrix.T
    return result[:, :2] / result[:, 2, None]


def backwards_mapping(image, output_shape, transformation, background=0):
    """Warp source to output using a FORWARD source->destination affine matrix.

    cv2.warpAffine internally performs inverse sampling; DO NOT invert the
    source->destination matrix before calling this function.
    """
    h, w = map(int, output_shape[:2])
    matrix = np.asarray(transformation, dtype=float)
    if matrix.shape == (3, 3):
        matrix = matrix[:2, :]
    if matrix.shape != (2, 3):
        raise ValueError('Expected 3x3 or 2x3 affine matrix')
    return cv2.warpAffine(np.asarray(image), matrix, (w, h),
                          flags=cv2.INTER_NEAREST,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=background)


def downsample_bilinear(img, factor):
    """Resize a 2D image by a positive real factor using bilinear interpolation."""
    img = np.asarray(img)
    if img.ndim != 2 or not np.isfinite(factor) or factor <= 0:
        raise ValueError('Expected 2D image and positive finite factor')
    h, w = img.shape
    new_size = (max(1, int(np.floor(w / factor))), max(1, int(np.floor(h / factor))))
    return cv2.resize(img, new_size, interpolation=cv2.INTER_LINEAR)


def histogram_equalization(img):
    """Histogram equalization for grayscale uint8; preserve original dtype where possible."""
    arr = np.asarray(img)
    if arr.ndim != 2:
        raise ValueError('Expected a 2D grayscale image')
    if arr.dtype == np.uint8:
        return cv2.equalizeHist(arr)
    values = arr.astype(np.float64)
    vmin, vmax = np.min(values), np.max(values)
    if vmax == vmin:
        return arr.copy()
    normalized = np.clip((values-vmin)*255/(vmax-vmin), 0, 255).astype(np.uint8)
    return cv2.equalizeHist(normalized)  # uint8 result for non-uint8 input


def convolve2d(img, kernel):
    """Zero-padded 2D convolution, output shape matches input."""
    a = np.asarray(img, dtype=np.float64)
    k = np.asarray(kernel, dtype=np.float64)
    if a.ndim != 2 or k.ndim != 2 or any(s % 2 != 1 for s in k.shape):
        raise ValueError('Expected 2D image and odd-shaped 2D kernel')
    return cv2.filter2D(a, cv2.CV_64F, np.flip(k, axis=(0, 1)), borderType=cv2.BORDER_CONSTANT)


def sobel_filter(img):
    """Gradient magnitude using 3x3 Sobel derivatives."""
    arr = np.asarray(img, dtype=np.float64)
    gx = cv2.Sobel(arr, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(arr, cv2.CV_64F, 0, 1, ksize=3)
    return np.hypot(gx, gy)


def filter_wrapper_fn(img, mode, kernel_size=3, sigma=1.0):
    """Apply mean, median, Gaussian or Sobel filtering to a grayscale image."""
    arr = np.asarray(img)
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError('kernel_size must be positive and odd')
    mode = mode.lower()
    if mode == 'mean':
        return cv2.blur(arr, (kernel_size, kernel_size))
    if mode == 'median':
        if arr.dtype not in (np.uint8, np.uint16, np.float32):
            return cv2.medianBlur(arr.astype(np.float32), kernel_size)
        return cv2.medianBlur(arr, kernel_size)
    if mode == 'gaussian':
        return gaussian_filter(arr, sigma=sigma)
    if mode == 'sobel':
        return sobel_filter(arr)
    raise ValueError('mode must be mean, median, gaussian, or sobel')


def get_keypoints(image, filtering=True, sigma=3):
    """Return OpenCV SIFT keypoints and Nx128 descriptors from a grayscale image."""
    arr = np.asarray(image)
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    if filtering:
        arr = gaussian_filter(arr, sigma=sigma)
    if arr.dtype != np.uint8:
        arr = np.asarray(arr, dtype=np.float64)
        lo, hi = np.min(arr), np.max(arr)
        arr = np.zeros(arr.shape, np.uint8) if hi <= lo else np.uint8(np.clip((arr-lo)*255/(hi-lo), 0, 255))
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(arr, None)
    if descriptors is None:
        descriptors = np.empty((0, 128), dtype=np.float32)
    return keypoints, descriptors


def intersect2d(array1, array2):
    """Return unique 2-column rows of array1 also found in array2."""
    a, b = np.asarray(array1), np.asarray(array2)
    if len(a) == 0 or len(b) == 0:
        return np.empty((0, 2), dtype=int)
    common = set(map(tuple, b.tolist()))
    return np.asarray([x for x in a if tuple(x) in common], dtype=int).reshape(-1, 2)


def matching(descriptors_1, descriptors_2, max_ratio=0.7, cross_checking=True):
    """Return Nx2 descriptor index pairs using Lowe's ratio and optional mutual check."""
    d1 = np.asarray(descriptors_1, dtype=np.float32)
    d2 = np.asarray(descriptors_2, dtype=np.float32)
    if d1.ndim != 2 or d2.ndim != 2 or d1.shape[1] != d2.shape[1] or len(d1) == 0 or len(d2) < 2:
        return np.empty((0, 2), dtype=int)
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    candidate = matcher.knnMatch(d1, d2, k=2)
    forward = np.array([(m.queryIdx, m.trainIdx) for pair in candidate if len(pair) == 2
                        for m,n in [pair] if m.distance < max_ratio*n.distance], dtype=int).reshape(-1, 2)
    if not cross_checking or len(forward) == 0:
        return forward
    if len(d1) < 2:
        return np.empty((0, 2), dtype=int)
    reverse_candidates = matcher.knnMatch(d2, d1, k=2)
    reverse = {(m.trainIdx, m.queryIdx) for pair in reverse_candidates if len(pair)==2
               for m,n in [pair] if m.distance < max_ratio*n.distance}
    return np.array([pair for pair in forward if tuple(pair) in reverse], dtype=int).reshape(-1,2)


def ransac(points_in, points_out, matches, percentage_outliers=0.5, probability=0.99, cutoff=20, k=3):
    """Robust affine match filtering; returns (inlier matches, number of inliers).

    Works with Nx2 point arrays in the same coordinate convention. 'k' and
    'percentage_outliers' are retained for notebook compatibility; OpenCV uses
    its internal minimal affine sampling and dynamically determined iterations.
    """
    matches = np.asarray(matches, dtype=int).reshape(-1, 2)
    if len(matches) < 3:
        return np.empty((0, 2), dtype=int), 0
    src = np.asarray(points_in, dtype=np.float32)[matches[:, 0]]
    dst = np.asarray(points_out, dtype=np.float32)[matches[:, 1]]
    matrix, inlier_mask = cv2.estimateAffine2D(src, dst, method=cv2.RANSAC,
                                              ransacReprojThreshold=float(cutoff),
                                              confidence=float(probability),
                                              maxIters=5000)
    if matrix is None or inlier_mask is None:
        return np.empty((0, 2), dtype=int), 0
    good = matches[inlier_mask.ravel().astype(bool)]
    return good, len(good)


def helper_plot_fn(img1, img2, transformed_img2):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, arr, title in zip(axes, [img1, img2, transformed_img2],
                              ['Source Image', 'Destination Image', 'Transformed Image']):
        ax.imshow(arr, cmap='gray')
        ax.set_title(title)
    plt.tight_layout()
    plt.show()
