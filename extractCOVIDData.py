import os
import sys
import pandas as pd
import shutil
from tqdm import tqdm


base_input_dir=os.path.expanduser('~/Downloads/COVID-19_Radiography_Dataset')
output_path=os.path.join(base_input_dir, 'processed')

classes = {
    'COVID': 0,
    'Pneumonia': 1,
    'Normal': 2,
}

num_images = {
    'COVID': 1000,
    'Pneumonia': 500,
    'Normal': 678,
}


def main():
    os.makedirs(os.path.join(output_path, 'images'), exist_ok=True)

    metadata = createMetadataFrame()
    metadata.to_csv(os.path.join(output_path, 'metadata.csv'), index=False)

    for index, row in metadata.iterrows():
        currentPath = os.path.join(base_input_dir, row['class'], row['filename']+'.png')
        newPath = os.path.join(output_path, 'images', row['filename']+'.png')
        shutil.copy(currentPath, newPath)

    # allImagePaths = getAllImagePaths()
    # # Remove all data with multiple labels
    # metadata = pd.read_csv(csv_file)
    # metadata = removeMultipleLabelData(metadata)
    # # Get class-separated, 'num_images_per_class' image paths
    # imagePaths = getImagePaths(allImagePaths, metadata)
    # writeProcessedData(imagePaths, metadata)


def createMetadataFrame():
    metadata = []
    columns = ["filename", "label", "class"]
    for c in classes:
        class_metadata = pd.read_csv(os.path.join(base_input_dir, f"{c}_metadata.csv"))
        i = 0
        for index, row in class_metadata.iterrows():
            if i >= num_images[c]:
                break
            metadata.append([row['FILE NAME'], classes[c], c])
            i += 1
    return pd.DataFrame(metadata, columns=columns)


if __name__=='__main__':
    main()
