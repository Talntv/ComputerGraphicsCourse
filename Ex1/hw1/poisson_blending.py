import cv2
import numpy as np
from scipy.sparse.linalg import spsolve
import argparse
from scipy.sparse import diags

def format_imgs(src, tgt, mask):
    src_shape = src.shape[:2]
    tgt_shape = tgt.shape[:2]
    mask_shape = mask.shape[:2]
    rows_src = tgt_shape[0] - src_shape[0]
    cols_src = tgt_shape[1] - src_shape[1]
    rows_mask = tgt_shape[0] - mask_shape[0]
    cols_mask = tgt_shape[1] - mask_shape[1]

    assert not (rows_src < 0 or cols_src < 0 or rows_mask < 0 or cols_mask < 0), "Incompatible images sizes"

    # Extend src_img & mask to tgt_img dimensions, fill with 0s around to center them on the target.
    new_mask = np.zeros(tgt.shape[:2])
    new_source = np.ones(tgt.shape)
    new_source[rows_src//2:rows_src//2+src_shape[0], cols_src//2:cols_src//2+src_shape[1]] = src
    new_mask[rows_mask//2:rows_mask//2+mask_shape[0], cols_mask//2:cols_mask//2+mask_shape[1]] = mask
    return new_source, new_mask

def get_B_matrices(laplacian, im_src, im_tgt, im_mask):
    # For the construction of the B matrix we referred to this article: https://hingxyu.medium.com/gradient-domain-fusion-using-poisson-blending-8a7dc1bbaa7b
    # We split the img by channels, as recommended
    Bs = []
    flat_mask = im_mask.flatten()
    for i in range(3):
        i_im_src = im_src[:, :, i].flatten()
        i_im_tgt = im_tgt[:, :, i].flatten()
        i_B = laplacian.dot(i_im_src)
        i_B[np.logical_not(flat_mask)] = i_im_tgt[np.logical_not(flat_mask)]
        Bs.append(i_B)
    return Bs

def poisson_blend(im_src, im_tgt, im_mask, center):
    im_src, im_mask = format_imgs(im_src, im_tgt, im_mask)
    global m, n
    m, n = im_tgt.shape[:2]
    img_size = m*n
    # Build the sparse laplacian matrix
    laplacian = -1*(diags([-4*np.ones(m*n), np.ones(m*n-1), np.ones(m*n-1), np.ones(m*n-n), np.ones(m*n-n)],
                      [0, -1, 1, -n, n], format='lil'))
    A = laplacian.copy()
    im_blend = im_tgt.copy()
    original_mask = im_mask.copy()
    
    # The next two lines are relevant for the case of the 'simple approach' from Oren's clarification
    # Bs = get_B_matrices(laplacian, im_src, im_tgt, im_mask)
    # flat_tgt = im_tgt.reshape(m*n, 3)

    for pixel in range(img_size):
        i = pixel // n
        j = pixel % n
        nearby_border = []
        neighbors = [(i-1,j), (i+1,j), (i,j-1), (i,j+1)]
        for row, col in neighbors:
            # Exclude out of boundaries neighbors, get border neighbors
            if (0 <= row < original_mask.shape[0] and
                0 <= col < original_mask.shape[1] and
                not original_mask[row, col]):
                nearby_border.append(n*row + col)
        if original_mask[i][j] == 0:
            # BG pixels
            A.rows[pixel] = [pixel]
            A.data[pixel] = [1]
        elif nearby_border:
            # Border pixles in the BG
            for border in nearby_border:
                # Implementation of the 'difficult approach' as described in Oren's clarification
                idx = A.rows[pixel].index(border)
                del A.rows[pixel][idx]
                del A.data[pixel][idx]

                # The next lines are the implementation of the 'simple approach', 
                # Here we update A and B as we go instead of removing outer border pixels from A
                # A.data[pixel][idx] = -1
                # for i in range(3):
                #     Bs[i][border] = flat_tgt[border][i]
            # This would make the inner border pixels to take the tgt_img colors
            im_mask[i][j] = 0
    # Compress A for improved performance
    A = A.tocsc()
    Bs = get_B_matrices(laplacian, im_src, im_tgt, im_mask)
    Xs = [np.clip(spsolve(A, b).reshape((m, n)), 0, 255) for b in Bs]
    im_blend[:, :, :] = np.array([Xs[0], Xs[1], Xs[2]]).transpose((1, 2, 0))
    return im_blend


def parse(imgname, bgname):
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_path', type=str, default=f'./data/imgs/{imgname}.jpg', help='image file path')
    parser.add_argument('--mask_path', type=str, default=f'./data/seg_GT/{imgname}.bmp', help='mask file path')
    parser.add_argument('--tgt_path', type=str, default=f'./data/bg/{bgname}.jpeg', help='mask file path')
    return parser.parse_args()

if __name__ == "__main__":
    # Load the source and target images
    imgname = 'llama'
    bgname = 'grass_mountains'
    args = parse(imgname, bgname)

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
