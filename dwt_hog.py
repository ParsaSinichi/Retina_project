import cv2
import numpy as np
import os
from natsort import natsorted
import pyfeats
from tqdm import tqdm
import multiprocessing as mp

def lbp(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))
    feat_list = []
    # using each R,G,B channel 
    for i in range(3):
        feat, _ = pyfeats.swt_features(image[:, :, i])
        # feat,_=pyfeats.hog_features(image[:,:,i])
        feat_list.append(feat)
    feat_list = np.concatenate(feat_list)
    return feat_list

def process_image(path):
    feature = lbp(path)
    label = 0 if "Non-AMD" in path else 1
    return feature, label

def extract_features(dataset_type):
    root_dir = f'G:\\AMD\\AMD_dataset(CLAHE)\\{dataset_type}'
    image_pathes = []

    # reading all of image paths
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_pathes.append(os.path.join(subdir, file))

    image_pathes = natsorted(image_pathes)
    num_processes = mp.cpu_count()

    with mp.Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.imap(process_image, image_pathes), total=len(image_pathes)))

    # Unpack the results
    feature_list, y = zip(*results)
    feature_list = list(feature_list)
    y = list(y)

    # saving features
    np.save(f"x_{dataset_type.lower()}_swt.npy", feature_list)
    np.save(f"y_{dataset_type.lower()}_swt.npy", y)

    print(f"{dataset_type} data extraction done.")

if __name__ == "__main__":
    for dataset_type in ["Train", "Validation", "Test"]:
        extract_features(dataset_type)
