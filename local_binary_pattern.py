import cv2
import numpy as np
from skimage.feature import local_binary_pattern
import os
from natsort import natsorted
from tqdm import tqdm
import multiprocessing as mp

def lbp(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))
    feat_list = []

    # using each r,g,b channel 
    for i in range(3):
        radius = 1  # radius of circle
        n_points = 8 * radius  # number of points to consider

        lbp = local_binary_pattern(image[:, :, i], n_points, radius, method='uniform')
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))  # 10 histogram bins 
        lbp_hist = lbp_hist.astype('float')
        lbp_hist /= lbp_hist.sum()
        feat_list.append(lbp_hist)

    feat_list = np.concatenate(feat_list)
    return feat_list

def process_image(path):
    feature = lbp(path)
    label = 0 if "Non-AMD" in path else 1
    return feature, label

def extract_features(dataset_type):
    root_dir = f'G:\\AMD\\AMD_dataset(CLAHE)\\{dataset_type}'
    image_pathes = []

    # image paths 
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                image_pathes.append(os.path.join(subdir, file))

    image_pathes = natsorted(image_pathes)
    num_processes = mp.cpu_count()

    #  pool of worker processes
    with mp.Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.imap(process_image, image_pathes), total=len(image_pathes)))

    # Unpack the results
    feature_list, y = zip(*results)
    feature_list = list(feature_list)
    y = list(y)

    np.save(f"x_{dataset_type.lower()}_LBP.npy", feature_list)
    np.save(f"y_{dataset_type.lower()}_LBP.npy", y)
    print(f"{dataset_type} data extraction done.")

if __name__ == "__main__":
    for dataset_type in ["Train", "Validation", "Test"]:
        extract_features(dataset_type)
