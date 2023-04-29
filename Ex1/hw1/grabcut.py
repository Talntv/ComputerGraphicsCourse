import time
import numpy as np
import cv2
import argparse
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import igraph as ig

GC_BGD = 0 # Hard bg pixel
GC_FGD = 1 # Hard fg pixel, will not be used
GC_PR_BGD = 2 # Soft bg pixel
GC_PR_FGD = 3 # Soft fg pixel

# Global variables
flat_img = np.float32()
n_link_capacities = []
n_sums = []
neighbors_of_each_pixel = {}
k = -1
num_of_pixels = -1
convergence = -1
latest_energy = -1

def timing_val(func):
    def wrapper(*arg, **kw):
        t1 = time.time()
        res = func(*arg, **kw)
        t2 = time.time()
        print(f'{func.__name__} took {(t2 - t1)}')
        return res
    return wrapper

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
    # Form a dictionary, where each key is a pixel, and its value is list of its neighbors
    for row in range(pixels.shape[0]):
        for col in range(pixels.shape[1]):
            edges = [pixels[i][j] for j in range(col-1, col+2) for i in range (row-1, row+2) if i >= 0 and i < len(pixels) and j >= 0 and j < len(pixels[0]) and not (i == row and j == col)]
            neighbors_of_each_pixel[pixels[row][col]] = edges
            total_number_of_neighbors += len(edges)
    beta = calculate_beta(neighbors_of_each_pixel, total_number_of_neighbors)
    
    global n_sums
    n_sums = np.zeros(flat_img.shape[0])
    # For every pixel and its neighbors, calculate the n_link_capacities
    for pixel, neighbors in enumerate(neighbors_of_each_pixel.values()):
        delta_new = (flat_img[pixel] - flat_img[neighbors])
        N_new = (50 / (np.linalg.norm(pixel - np.array(neighbors).reshape(1,len(neighbors)), axis=0)) * np.exp(-beta * np.diag(delta_new.dot(delta_new.T))))
        n_link_capacities.extend(N_new)
        n_sums[pixel] += np.sum(N_new)
    return n_sums

def get_T_links_capacities(mask, bgGMM, fgGMM):
    mask = mask.reshape(-1)
    K = max(n_sums)
    source_capacities = np.where((mask == GC_FGD), K, 0)
    target_capacities = np.where(mask == GC_BGD, K, 0)
    bgDn = calculate_Dn(bgGMM, flat_img[(mask != GC_BGD) & (mask != GC_FGD)])
    fgDn = calculate_Dn(fgGMM, flat_img[(mask != GC_BGD) & (mask != GC_FGD)])
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
        distance_from_mean = (z - mean)
        # the np.einsum enables inplace calculation to reduce time and memory costs
        inner = np.einsum('ij,ij->i', distance_from_mean, np.dot(np.linalg.inv(covariance_matrix), distance_from_mean.T).T)
        pdf = (pi / np.sqrt(det)) * np.exp(-0.5 * inner)
        d += pdf
    return -1 * np.log(d)

def update_GMM(GMM : GaussianMixture, img, pixels):
    n_components=GMM.n_components
    kmeans = KMeans(n_clusters=n_components, n_init='auto')
    kmeans.fit(pixels)
    centroids = kmeans.cluster_centers_
    covariances = np.zeros((n_components, img.shape[-1], img.shape[-1]))
    for i in range(n_components):
        covariances[i] = 0.01 * np.eye(img.shape[-1])
    weights = np.zeros(n_components)
    for i in range(n_components):
        weights[i] = np.sum(kmeans.labels_ == i) / len(pixels)
    for i in range(n_components):
        indices = np.where(kmeans.labels_ == i)[0]
        if indices.size > 0:
            X = pixels[indices]
            N = len(indices)
            mean = centroids[i]
            covariance = (1 / N) * np.dot((X - mean).T, (X - mean))
            covariances[i] = covariance
    GMM.means_ = centroids
    GMM.covariances_ = covariances
    GMM.weights_ = weights   
    # precisions = np.zeros((n_components, img.shape[-1], img.shape[-1]))
    # for i in range(n_components):
    #     # Handle singular matrices
    #     if np.linalg.det(GMM.covariances_[i]) == 0:
    #         precisions[i] = GMM.covariances_[i] + np.eye(3) * 0.001
    #     else:
    #         precisions[i] = np.linalg.inv(GMM.covariances_[i])

# Define the GrabCut algorithm function
@timing_val
def grabcut(img, rect, n_iter=5):
    # Assign initial labels to the pixels based on the bounding box
    img = np.asarray(img, dtype=np.float64)
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    mask.fill(GC_BGD)
    x, y, w, h = rect
    w -= x
    h -= y

    #Initalize the inner square to Foreground
    mask[y:y+h, x:x+w] = GC_PR_FGD
    mask[(rect[1]+rect[3])//2, (rect[0]+rect[2])//2] = GC_FGD
    bgGMM, fgGMM = initialize_GMMs(img)
    for i in range(n_iter):
        print(f'Performing the {i+1}th iteration')
        bgGMM, fgGMM = update_GMMs(img, mask, bgGMM, fgGMM)
        mincut_sets, energy = calculate_mincut(img, mask, bgGMM, fgGMM)
        mask = update_mask(mincut_sets, mask)
        if check_convergence(energy):
            break

    print(f'Finished after {i+1} iterations')

    # Return the final mask and the GMMs
    return mask, bgGMM, fgGMM

def initialize_GMMs(img, n_components=5):
    bgGMM = GaussianMixture(n_components=n_components, covariance_type='full')
    fgGMM = GaussianMixture(n_components=n_components, covariance_type='full')

    global flat_img
    global num_of_pixels
    flat_img = np.float32(img).reshape(-1,3)
    h, w = img.shape[:2]
    num_of_pixels = h*w
    # Only happens once, so can be part of the initialization
    get_N_links_capacities(img)
    return bgGMM, fgGMM

def update_GMMs(img, mask, bgGMM : GaussianMixture, fgGMM: GaussianMixture):
    bg_pixels = img[np.logical_or(mask == GC_BGD, mask == GC_PR_BGD)]
    fg_pixels = img[np.logical_or(mask == GC_PR_FGD, mask == GC_FGD)]
    update_GMM(bgGMM, img, bg_pixels)
    update_GMM(fgGMM, img, fg_pixels)
    return bgGMM, fgGMM

def calculate_mincut(img, mask, bgGMM, fgGMM):
    source_capacities, target_capacities = get_T_links_capacities(mask, bgGMM, fgGMM)
    capacities = source_capacities + target_capacities + n_link_capacities
    # Each vertex represents a pixel, +2 additional vertices for source and target
    g = ig.Graph(num_of_pixels + 2)
    source, target = num_of_pixels, num_of_pixels + 1
    neighbor_edges = [(pixel, neighbor) for pixel in neighbors_of_each_pixel for neighbor in neighbors_of_each_pixel[pixel]]
    source_edges = np.column_stack((np.full(num_of_pixels, source), np.arange(num_of_pixels)))
    target_edges = np.column_stack((np.arange(num_of_pixels), np.full(num_of_pixels, target)))
    # Every pixel is attached to the source, the sink, and to all its neighbors
    g.add_edges(np.concatenate((source_edges, target_edges)).tolist() + neighbor_edges)
    min_cut = g.st_mincut(source, target, capacities)
    energy = min_cut.value
    return [min_cut.partition[0], min_cut.partition[1]], energy

def update_mask(mincut_sets, mask: np.ndarray):
    rows, columns = mask.shape
    fg_v = mincut_sets[0]
    pr_indexes = np.where(np.logical_or(mask == GC_PR_BGD, mask == GC_PR_FGD))
    img_indexes = np.arange(rows * columns, dtype=np.uint32).reshape(rows, columns)
    # Every mask pixel which wasn't a hard pixel is reassigned a value, based on the latest mincut
    mask[pr_indexes] = np.where(np.isin(img_indexes[pr_indexes], fg_v), GC_PR_FGD, GC_PR_BGD)
    return mask

def check_convergence(energy):
    global latest_energy
    # We found 800 to correlate well with not that many pixels moving from FG to BG or the otherway around.
    converged = np.abs(latest_energy - energy) < 800
    latest_energy = energy
    return converged

def cal_metric(predicted_mask, gt_mask):
    correct_pixels = np.sum(predicted_mask == gt_mask)
    total_pixels = predicted_mask.shape[0] * predicted_mask.shape[1]
    accuracy = correct_pixels / total_pixels

    intersection = np.sum(predicted_mask & gt_mask)
    union = np.sum(predicted_mask | gt_mask)
    jaccard_similarity = intersection / union

    return accuracy, jaccard_similarity

def parse(imgname):
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_name', type=str, default=imgname, help='name of image from the course files')
    parser.add_argument('--eval', type=int, default=1, help='calculate the metrics')
    parser.add_argument('--input_img_path', type=str, default='', help='if you wish to use your own img_path')
    parser.add_argument('--use_file_rect', type=int, default=1, help='Read rect from course files')
    parser.add_argument('--rect', type=str, default='1,1,100,100', help='if you wish change the rect (x,y,w,h')
    return parser.parse_args()

if __name__ == '__main__':
    # Load an example image and define a bounding box around the object of interest
    imgname = 'teddy'
    args = parse(imgname)

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
    # All pixels classifed as BG are given HARD BG value.
    mask[mask == GC_PR_BGD] = GC_BGD
    mask = cv2.threshold(mask, 0, 1, cv2.THRESH_BINARY)[1]

    # Print metrics only if requested (valid only for course files)
    if args.eval:
        gt_mask = cv2.imread(f'data/seg_GT/{args.input_name}.bmp', cv2.IMREAD_GRAYSCALE)
        gt_mask = cv2.threshold(gt_mask, 0, 1, cv2.THRESH_BINARY)[1]
        acc, jac = cal_metric(mask, gt_mask)
        print(f'Accuracy={acc}, Jaccard={jac}')

    img_cut = img * (mask[:, :, np.newaxis])

    cv2.imwrite(f'{imgname}_mask_cut.bmp', 255 * mask)
    cv2.imwrite(f'{imgname}_imt_cut.bmp', img_cut)
    cv2.imshow('Original Image', img)
    cv2.imshow('GrabCut Mask', 255 * mask)
    cv2.imshow('GrabCut Result', img_cut)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
