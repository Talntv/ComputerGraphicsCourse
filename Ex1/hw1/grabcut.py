import time
import numpy as np
import cv2
import argparse
from sklearn.mixture import GaussianMixture
import igraph as ig

GC_BGD = 0 # Hard bg pixel
GC_FGD = 1 # Hard fg pixel, will not be used
GC_PR_BGD = 2 # Soft bg pixel
GC_PR_FGD = 3 # Soft fg pixel

def timing_val(func):
    def wrapper(*arg, **kw):
        t1 = time.time()
        res = func(*arg, **kw)
        t2 = time.time()
        print(f'{func.__name__} took {(t2 - t1)}')
        return res
    return wrapper

flat_img = np.float32()
n_link_capacities = []
num_of_pixels = -1
n_sums = []
neighbors_of_each_pixel = {}
k = -1

@timing_val
def calculate_beta(neighbors_of_each_pixel: dict, total_number_of_neighbors):
    sum_dist = 0
    for pixel in neighbors_of_each_pixel:
        neighbors_arr = np.array(neighbors_of_each_pixel[pixel])
        dist = flat_img[pixel] - flat_img[neighbors_arr]
        sum_dist += np.sum(dist * dist)
    return 1 / (2 * sum_dist / total_number_of_neighbors)

@timing_val
def get_N_links_capacities(img):
    global n_link_capacities
    global neighbors_of_each_pixel
    pixels = np.arange(num_of_pixels).reshape(img.shape[:-1])
    total_number_of_neighbors = 0
    for row in range(pixels.shape[0]):
        for col in range(pixels.shape[1]):
            edges = [pixels[i][j] for j in range(col-1, col+2) for i in range (row-1, row+2) if i >= 0 and i < len(pixels) and j >= 0 and j < len(pixels[0]) and not (i == row and j == col)]
            neighbors_of_each_pixel[pixels[row][col]] = edges
            total_number_of_neighbors += len(edges)

    beta = calculate_beta(neighbors_of_each_pixel, total_number_of_neighbors)
    print(beta)
    global n_sums
    n_sums = np.zeros(flat_img.shape[0])
    for pixel in neighbors_of_each_pixel:
        for neighbor in neighbors_of_each_pixel[pixel]:
            delta = (flat_img[pixel] - flat_img[neighbor])
            N = (50 / (np.linalg.norm(pixel - neighbor)) * np.exp(-beta * delta.dot(delta)))
            n_link_capacities.append(N)
            n_sums[pixel] += N
    return n_sums

@timing_val
def get_T_links_capacities(mask, bgGMM, fgGMM):
    mask = mask.reshape(-1)
    K = max(n_sums)
    source_capacities = np.where((mask == GC_FGD), K, 0)
    target_capacities = np.where(mask == GC_BGD, K, 0)
    bgDn = np.apply_along_axis(lambda x: calculate_Dn(bgGMM, x), axis=1, arr=flat_img[(mask != GC_BGD) & (mask != GC_FGD)])
    fgDn = np.apply_along_axis(lambda x: calculate_Dn(fgGMM, x), axis=1, arr=flat_img[(mask != GC_BGD) & (mask != GC_FGD)])
    print(f'source {len(source_capacities)} and target {len(target_capacities)}')
    
    source_capacities[(mask != GC_BGD) & (mask != GC_FGD)] = bgDn
    target_capacities[(mask != GC_BGD) & (mask != GC_FGD)] = fgDn
    
    return source_capacities.tolist(), target_capacities.tolist()

def calculate_Dn(GMM : GaussianMixture, z):
    d = 0
    for i in range(GMM.n_components):
        pi = GMM.weights_[i]
        mean = GMM.means_[i]
        covariance_matrix = GMM.covariances_[i]
        det = np.linalg.det(covariance_matrix)
        pdf = (pi / np.sqrt(det)) * np.exp(-0.5 * np.dot(np.dot((z - mean).T, np.linalg.inv(covariance_matrix)), (z - mean)))
        d += pdf
    return -1 * np.log(d)

# Define the GrabCut algorithm function
@timing_val
def grabcut(img, rect, n_iter=5):
    # Assign initial labels to the pixels based on the bounding box
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    mask.fill(GC_BGD)
    x, y, w, h = rect

    w -= x
    h -= y

    #Initalize the inner square to Foreground
    mask[y:y+h, x:x+w] = GC_PR_FGD
    mask[rect[1]+rect[3]//2, rect[0]+rect[2]//2] = GC_FGD

    bgGMM, fgGMM = initalize_GMMs(img, mask)

    num_iters = 1
    for i in range(num_iters):
        #Update GMM
        bgGMM, fgGMM = update_GMMs(img, mask, bgGMM, fgGMM)

        mincut_sets, energy = calculate_mincut(img, mask, bgGMM, fgGMM)

        mask = update_mask(mincut_sets, mask)

        if check_convergence(energy):
            break

    # Return the final mask and the GMMs
    return mask, bgGMM, fgGMM


@timing_val
def initalize_GMMs(img, mask, n_components=5):
    bg_pixels = img[np.logical_or(mask == GC_BGD, mask == GC_PR_BGD)]
    fg_pixels = img[np.logical_or(mask == GC_PR_FGD, mask == GC_FGD)]    

    bgGMM = GaussianMixture(n_components=n_components, covariance_type='full', init_params='kmeans')
    bgGMM.fit(bg_pixels)
    fgGMM = GaussianMixture(n_components=n_components, covariance_type='full', init_params='kmeans')
    fgGMM.fit(fg_pixels)
    # TODO: use bgGMM.set_params() to update instead of 'fit'

    global flat_img
    flat_img = np.float32(img).reshape(-1,3)

    h, w = img.shape[:2]
    global num_of_pixels
    num_of_pixels = h*w
    get_N_links_capacities(img)

    return bgGMM, fgGMM


@timing_val
def update_GMMs(img, mask, bgGMM : GaussianMixture, fgGMM: GaussianMixture):
    # # Reshape img and mask to 2D arrays
    # img_2d = img.reshape((-1, img.shape[-1]))
    # mask_2d = mask.ravel()

    # # Compute the likelihoods of each pixel belonging to each Gaussian component in the background and foreground GMMs
    # bg_likelihoods = bgGMM.predict_proba(img_2d)
    # fg_likelihoods = fgGMM.predict_proba(img_2d)
    # mask_2d = mask_2d.reshape(bg_likelihoods.shape[0], -1)

    # # Compute the responsibilities of each pixel for each Gaussian component in the background and foreground GMMs
    # bg_responsibilities = (1 - mask_2d) * bg_likelihoods
    # fg_responsibilities = mask_2d * fg_likelihoods

    # # Normalize the responsibilities
    # total_responsibilities = bg_responsibilities + fg_responsibilities
    # bg_responsibilities /= total_responsibilities
    # fg_responsibilities /= total_responsibilities

    # # Compute the total responsibilities for each Gaussian component
    # bg_total_responsibilities = np.sum(bg_responsibilities, axis=0)
    # fg_total_responsibilities = np.sum(fg_responsibilities, axis=0)

    # # Update the means, covariances, determinants, and mixing weights of each Gaussian component
    # for i in range(bgGMM.n_components):
    #     # Compute the new mean and covariance for the background GMM
    #     indices = np.where(bg_responsibilities[:, i] > 0)[0]
    #     if indices.size > 0:
    #         bgGMM.means_[i] = np.mean(img_2d[indices], axis=0)
    #         bgGMM.covariances_[i] = np.cov(img_2d[indices], rowvar=False, bias=True) * ((len(img_2d[indices]) -1) / len(img_2d[indices]))
    #         bgGMM.weights_[i] = bg_total_responsibilities[i] / img_2d.shape[0]
    
    # print(f'fgGMM covs before update: {fgGMM.covariances_}')
    # for i in range(fgGMM.n_components):
    #     # Compute the new mean and covariance for the foreground GMM
    #     indices = np.where(fg_responsibilities[:, i] > 0)[0]
    #     if indices.size > 0:
    #         fgGMM.means_[i] = np.mean(img_2d[indices], axis=0)
    #         fgGMM.covariances_[i] = np.cov(img_2d[indices], rowvar=False, bias=True)
    #         fgGMM.weights_[i] = fg_total_responsibilities[i] / img_2d.shape[0]
    # print(f'\nfgGMM covs after update: {fgGMM.covariances_}')

    # return bgGMM, fgGMM

    bgPixels = np.where(np.logical_or(mask == GC_PR_BGD, mask == GC_BGD))
    fgPixels = np.where(np.logical_or(mask == GC_PR_FGD, mask == GC_FGD))
    print(f'fgGMM covs before update: {fgGMM.covariances_}')

    bgSamples = img[bgPixels]
    fgSamples = img[fgPixels]
    bgGMM.fit(bgSamples)
    fgGMM.fit(fgSamples)
    print(f'fgGMM covs after update: {fgGMM.covariances_}')

    # TODO: use bgGMM.set_params() to update instead of 'fit'

    return bgGMM, fgGMM

@timing_val
def calculate_mincut(img, mask, bgGMM, fgGMM):
    source_capacities, target_capacities = get_T_links_capacities(mask, bgGMM, fgGMM)
    capacities = source_capacities + target_capacities + n_link_capacities
    g = ig.Graph(num_of_pixels + 2)
    source, target = num_of_pixels, num_of_pixels + 1
    print('initializied graph')
    neighbor_edges = []
    for pixel in neighbors_of_each_pixel:
        for neighbor in neighbors_of_each_pixel[pixel]:
            neighbor_edges.append((pixel, neighbor))

    source_edges = [(source, i) for i in range(num_of_pixels)]
    target_edges = [(i, target) for i in range(num_of_pixels)]
    e = source_edges + target_edges + neighbor_edges

    g.add_edges(e)
    min_cut = g.st_mincut(source, target, capacities)
    energy = min_cut.value
    print(f'energy is {energy}')
    return [min_cut.partition[0], min_cut.partition[1]], energy


@timing_val
def update_mask(mincut_sets, mask):
    rows, columns = mask.shape
    fg_v = mincut_sets[0]
    bg_v = mincut_sets[1]
    pr_indexes = np.where(np.logical_or(mask == GC_PR_BGD, mask == GC_PR_FGD))
    img_indexes = np.arange(rows * columns, dtype=np.uint32).reshape(rows, columns)
    mask[pr_indexes] = np.where(np.isin(img_indexes[pr_indexes], fg_v), GC_PR_FGD, GC_PR_BGD)
    return mask

@timing_val
def check_convergence(energy):
    return False

@timing_val
def cal_metric(predicted_mask, gt_mask):
    # TODO: implement metric calculation

    return 100, 100

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_name', type=str, default='banana1', help='name of image from the course files')
    parser.add_argument('--eval', type=int, default=1, help='calculate the metrics')
    parser.add_argument('--input_img_path', type=str, default='', help='if you wish to use your own img_path')
    parser.add_argument('--use_file_rect', type=int, default=1, help='Read rect from course files')
    parser.add_argument('--rect', type=str, default='1,1,100,100', help='if you wish change the rect (x,y,w,h')
    return parser.parse_args()

if __name__ == '__main__':
    # Load an example image and define a bounding box around the object of interest
    args = parse()


    if args.input_img_path == '':
        input_path = f'data/imgs/{args.input_name}.jpg'
    else:
        input_path = args.input_img_path

    if args.use_file_rect:
        rect = tuple(map(int, open(f"data/bboxes/{args.input_name}.txt", "r").read().split(' ')))
    else:
        rect = tuple(map(int,args.rect.split(',')))


    img = cv2.imread(input_path)

    # Run the GrabCut algorithm on the image and bounding box
    mask, bgGMM, fgGMM = grabcut(img, rect)
    mask[mask == GC_PR_BGD] = GC_BGD
    mask = cv2.threshold(mask, 0, 1, cv2.THRESH_BINARY)[1]

    # Print metrics only if requested (valid only for course files)
    if args.eval:
        gt_mask = cv2.imread(f'data/seg_GT/{args.input_name}.bmp', cv2.IMREAD_GRAYSCALE)
        gt_mask = cv2.threshold(gt_mask, 0, 1, cv2.THRESH_BINARY)[1]
        acc, jac = cal_metric(mask, gt_mask)
        print(f'Accuracy={acc}, Jaccard={jac}')

    # Apply the final mask to the input image and display the results
    img_cut = img * (mask[:, :, np.newaxis])
    cv2.imshow('Original Image', img)
    cv2.imshow('GrabCut Mask', 255 * mask)
    cv2.imshow('GrabCut Result', img_cut)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
