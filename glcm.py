import SimpleITK as sitk
from radiomics import featureextractor
from PIL import Image
import numpy as np
import cv2
import os
from natsort import natsorted
from tqdm import tqdm
import multiprocessing as mp

def extract_feat(image_path):
    image = Image.open(image_path)
    threshold_value = 5

    # Resize 
    new_size = (256, 256)
    image = image.resize(new_size, Image.ANTIALIAS)
    image_np = np.array(image)

    extractor = featureextractor.RadiomicsFeatureExtractor()
    all_features = []

    # using each R,G,B channel
    for channel in range(image_np.shape[2]):
        channel_data = image_np[:, :, channel]

        #  binary mask
        _, binary_mask = cv2.threshold(channel_data, threshold_value, 255, cv2.THRESH_BINARY)
        binary_mask = binary_mask // 255

        image_sitk = sitk.GetImageFromArray(channel_data)
        mask_sitk = sitk.GetImageFromArray(binary_mask.astype(np.uint8))

        # Extract features
        features = extractor.execute(image_sitk, mask_sitk)

        glcm = []
        glrlm = []
        gldm = []
        glszm = []
        firstorder = []

        for featureName in features.keys():
            if "glcm" in featureName:
                glcm.append(features[featureName])
            # if "glrlm" in featureName:
            #     glrlm.append(features[featureName])
            # if "gldm" in featureName:
            #     gldm.append(features[featureName])
            # if "glszm" in featureName:
            #     glszm.append(features[featureName])
            # if "firstorder" in featureName:
            #     firstorder.append(features[featureName])

        # Combine features
        channel_features = np.concatenate((glcm, glrlm, gldm, glszm, firstorder))
        all_features.append(channel_features)

    all_features = np.concatenate(all_features)
    return all_features

def process_image(path):
    feature = extract_feat(path)
    label = 0 if "Non-AMD" in path else 1
    return feature, label

def extract_features(dataset_type):
    root_dir = f'G:\\AMD\\AMD_dataset(CLAHE)\\{dataset_type}'
    image_pathes = []

    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_pathes.append(os.path.join(subdir, file))

    image_pathes = natsorted(image_pathes)
    num_processes = mp.cpu_count()

    with mp.Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.imap(process_image, image_pathes), total=len(image_pathes)))

    feature_list, y = zip(*results)
    feature_list = list(feature_list)
    y = list(y)

    # saving features
    np.save(f"x_{dataset_type.lower()}_glcm.npy", feature_list)
    np.save(f"y_{dataset_type.lower()}_glcm.npy", y)

    print(f"{dataset_type} data extraction done.")

if __name__ == "__main__":
    for dataset_type in ["Train", "Validation", "Test"]:
        extract_features(dataset_type)
