import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sps
from scipy.sparse.linalg import spsolve
import argparse
from scipy.ndimage import sobel
from scipy.sparse import diags

def poisson_blend(im_src, im_tgt, im_mask, center):
    m, n = im_src.shape[:2]
    # cut the target img to match the source img
    im_tgt = im_tgt[:m, :n]
    laplacian = diags([-4*np.ones(m*n), np.ones(m*n-1), np.ones(m*n-1), np.ones(m*n-m), np.ones(m*n-m)],
                      [0, -1, 1, -m, m])
    # Find the boundary of the binary mask using gradients
    mask = im_mask.astype(bool)
    mask_sobel = np.logical_or.reduce([sobel(mask, axis=i, mode='constant') for i in range(2)])
    boundary_indices = np.flatnonzero(mask_sobel)

    # These 4 lines copy the pixels defined in the mask in the source img, and paste them in the target
    masked_region = im_src.copy()
    masked_region[im_mask == 0] = 0
    im_tgt[im_mask != 0] = masked_region[im_mask != 0]
    cv2.imshow('Cloned image', im_tgt)

    # These 4 lines 'draw' the outline of the mask to the target img
    p = im_tgt.reshape(-1, 3)
    p[boundary_indices] = 0
    p = p.reshape((m, n, 3))
    cv2.imshow('Cloned image', p)

    # Set up the Poisson equation
    b = np.zeros_like(im_tgt)
    # x_center, y_center = center
    # x_offset = max(0, x_center - im_src.shape[0] // 2)
    # y_offset = max(0, y_center - im_src.shape[1] // 2)
    # x_min, x_max = x_offset, x_offset + im_src.shape[0]
    # y_min, y_max = y_offset, y_offset + im_src.shape[1]
    # b[x_min:x_max, y_min:y_max] = im_src - im_tgt[x_min:x_max, y_min:y_max]
    b_flat = b.reshape(-1, 3)
    laplacian_boundary = laplacian[boundary_indices][:, boundary_indices]

    # Solve the Poisson equation using sparse linear algebra
    x_flat = spsolve(laplacian_boundary, -b_flat[boundary_indices])

    # Blend the source and target images using the Poisson result
    x = im_tgt.reshape(-1, 3)
    x_flat_clipped = np.clip(x_flat, 0, 255)
    x_flat_rounded = np.round(x_flat_clipped).astype(np.float64)
    x[boundary_indices] = x_flat_rounded

    # Place the blended image onto the target image
    im_blend = x.reshape((m, n, 3))

    # Save the blended image
    return im_blend




def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_path', type=str, default='./data/imgs/banana2.jpg', help='image file path')
    parser.add_argument('--mask_path', type=str, default='./data/seg_GT/banana2.bmp', help='mask file path')
    parser.add_argument('--tgt_path', type=str, default='./data/bg/table.jpg', help='mask file path')
    return parser.parse_args()

if __name__ == "__main__":
    # Load the source and target images
    args = parse()

    im_tgt = cv2.imread(args.tgt_path, cv2.IMREAD_COLOR)
    im_src = cv2.imread(args.src_path, cv2.IMREAD_COLOR)
    if args.mask_path == '':
        im_mask = np.full(im_src.shape, 255, dtype=np.uint8)
    else:
        im_mask = cv2.imread(args.mask_path, cv2.IMREAD_GRAYSCALE)
        im_mask = cv2.threshold(im_mask, 0, 255, cv2.THRESH_BINARY)[1]

    center = (int(im_tgt.shape[1] / 2), int(im_tgt.shape[0] / 2))

    im_clone = poisson_blend(im_src, im_tgt, im_mask, center)

    cv2.imshow('Cloned image', im_clone)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
