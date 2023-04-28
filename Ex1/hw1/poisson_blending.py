import cv2
import numpy as np
from scipy.sparse.linalg import spsolve
import argparse
from scipy.sparse import diags

def format_imgs(im_source, im_target, im_mask):
    source_rows, source_columns = im_source.shape[:2]
    target_rows, target_columns = im_target.shape[:2]
    mask_rows, mask_columns = im_mask.shape[:2]

    rows_source = target_rows - source_rows
    cols_source = target_columns - source_columns
    rows_mask = target_rows - mask_rows
    cols_mask = target_columns - mask_columns

    assert not (rows_source < 0 or cols_source < 0 or rows_mask < 0 or cols_mask < 0), "Incompatible images sizes"

    new_mask = np.zeros(im_target.shape[:2])
    new_source = np.ones(im_target.shape)

    new_source[rows_source // 2:rows_source // 2 + source_rows, cols_source // 2:cols_source // 2 + source_columns] = im_source
    new_mask[rows_mask // 2:rows_mask // 2 + mask_rows, cols_mask // 2:cols_mask // 2 + mask_columns] = im_mask

    return new_source, new_mask

def get_zero_pixels(original_mask, i, j):
    zero_pixels = []
    # Upper neighbor
    if i == 0 or not original_mask[i - 1, j]:
        zero_pixels.append(n*(i-1)+j)
    # left neighbor
    if j == 0 or not original_mask[i, j - 1]:
        zero_pixels.append(n*i+j-1)
    # down neighbor
    if i + 1 == m or not original_mask[i + 1, j]:
        zero_pixels.append(n*(i+1)+j)
    # right neighbor
    if j + 1 == n or not original_mask[i, j + 1]:
        zero_pixels.append(n*i+j+1)
    return zero_pixels

def get_B(laplacian, im_src, im_tgt, im_mask):
    # For the construction of the B matrix we referred to this article: https://hingxyu.medium.com/gradient-domain-fusion-using-poisson-blending-8a7dc1bbaa7b
    R_im_src = im_src[:, :, 0].flatten()
    G_im_src = im_src[:, :, 1].flatten()
    B_im_src = im_src[:, :, 2].flatten()
    R_im_tgt = im_tgt[:, :, 0].flatten()
    G_im_tgt = im_tgt[:, :, 1].flatten()
    B_im_tgt = im_tgt[:, :, 2].flatten()

    _im_mask = im_mask.flatten()

    R_B = laplacian.dot(R_im_src)
    G_B = laplacian.dot(G_im_src)
    B_B = laplacian.dot(B_im_src)

    R_B[_im_mask == 0] = R_im_tgt[_im_mask == 0]
    G_B[_im_mask == 0] = G_im_tgt[_im_mask == 0]
    B_B[_im_mask == 0] = B_im_tgt[_im_mask == 0]

    return [R_B, G_B, B_B]

def poisson_blend(im_src, im_tgt, im_mask, center):
    im_src, im_mask = format_imgs(im_src, im_tgt, im_mask)
    global m, n
    m, n = im_tgt.shape[:2]
    img_size = m*n
    laplacian = -1*(diags([-4*np.ones(m*n), np.ones(m*n-1), np.ones(m*n-1), np.ones(m*n-n), np.ones(m*n-n)],
                      [0, -1, 1, -n, n], format='lil'))
    A = laplacian.copy()

    # If we wish to update A & B instead of deleting A border pixels, we need these lines
    # Bs = get_B(laplacian, im_src, im_tgt, im_mask)
    flat_tgt = im_tgt.reshape(m*n, 3)

    original_mask = im_mask.copy()
    for pixel in range(img_size):
        curr_row = pixel // n
        curr_col = pixel % n
        zero_pixels = get_zero_pixels(original_mask, curr_row, curr_col)
        if original_mask[curr_row][curr_col] == 0:
            # Border pixels within the FG
            A.rows[pixel] = [pixel]
            A.data[pixel] = [1]
        elif zero_pixels:
            # Border pixles in the BG
            for zero_pixel in zero_pixels:
                # Implementation of the 'difficult approach' as described in Oren's clarifition
                idx = A.rows[pixel].index(zero_pixel)
                del A.rows[pixel][idx]
                del A.data[pixel][idx]

                # Simple approach, update A and B instead of removing outer border pixels
                # A.data[pixel][idx] = -1
                # for i in range(3):
                #     Bs[i][zero_pixel] = flat_tgt[zero_pixel][i]

            im_mask[curr_row][curr_col] = 0
    A = A.tocsc()
            
    im_blend = im_tgt.copy()

    Bs = get_B(laplacian, im_src, im_tgt, im_mask)
    Xs = [np.clip(spsolve(A, b).reshape((m, n)), 0, 255) for b in Bs]
    im_blend[:, :, :] = np.array([Xs[0], Xs[1], Xs[2]]).transpose((1, 2, 0))

    return im_blend


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_path', type=str, default='./data/imgs/banana1.jpg', help='image file path')
    parser.add_argument('--mask_path', type=str, default='./data/seg_GT/banana1.bmp', help='mask file path')
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
