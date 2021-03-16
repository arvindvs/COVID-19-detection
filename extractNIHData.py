import os
import pandas as pd
import shutil
from tqdm import tqdm


csv_file=os.path.expanduser('~/Downloads/archive/Data_Entry_2017.csv')
output_path=os.path.expanduser('~/Downloads/archive/processed')
image_dir=os.path.expanduser('~/Downloads/archive/images')
num_images_per_class = 1000

# classes = {
#     'Atelectasis': 3,
#     'Consolidation': 4,
#     'Infiltration': 5,
#     'Pneumothorax': 6,
#     'Edema': 7,
#     'Emphysema': 8,
#     'Fibrosis': 9,
#     'Effusion': 10,
#     'Pneumonia': 1,
#     'Pleural_Thickening': 11,
#     'Cardiomegaly': 12,
#     'Nodule': 13,
#     'Mass': 14,
#     'Hernia': 15,
#     'No Finding': 2
# }

classes = {
    'Atelectasis': 3,
    'Consolidation': 4,
    'Infiltration': 5,
    'Pneumothorax': 6,
    'Effusion': 7,
    'Pneumonia': 1,
    'Pleural_Thickening': 8,
    'Cardiomegaly': 9,
    'Nodule': 10,
    'Mass': 11,
    'No Finding': 2
}


def main():
    allImagePaths = getAllImagePaths()
    # Remove all data with multiple labels
    metadata = pd.read_csv(csv_file)
    metadata = removeMultipleLabelData(metadata)
    # Get class-separated, 'num_images_per_class' image paths
    imagePaths = getImagePaths(allImagePaths, metadata)
    writeProcessedData(imagePaths, metadata)


def getAllImagePaths():
    paths = set()
    dirs = getSubdirectories(image_dir)
    for dir in dirs: # Iterate through subdirectories
        files = [os.path.join(dir, f) for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]
        paths.update(files)
    return paths


def removeMultipleLabelData(metadata):
    print(metadata.head())
    contains_one = metadata["Finding Labels"].str.contains('|', regex=False)
    metadata_single_label = metadata[~contains_one]
    print(metadata_single_label.head())
    print("Size of old metadata:", len(metadata))
    print("Size of new metadata:", len(metadata_single_label))
    print()
    return metadata_single_label


def getImagePaths(allImagePaths, metadata):
    imageNames = {c: [] for c in classes}
    dirs = getSubdirectories(image_dir)
    for c in classes:
        class_filter = metadata['Finding Labels'] == c
        filtered_metadata = metadata[class_filter]
        numImages = num_images_per_class if c!="No Finding" else num_images_per_class/2
        for index, row in filtered_metadata.iterrows():
            if (len(imageNames[c]) >= numImages):
                break
            name = row['Image Index']
            fullPath = getValidPath(allImagePaths, name)
            if (fullPath):
                imageNames[c].append(fullPath)
    for c, nameList in imageNames.items():
        print(c, len(nameList))
    return imageNames


def getValidPath(allImagePaths, name):
    dirs = getSubdirectories(image_dir)
    for dir in dirs:
        path = os.path.join(dir, name)
        if path in allImagePaths:
            return path
    return None


def getSubdirectories(path):
    return [os.path.join(path, d) for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]


def writeProcessedData(imagePaths, metadata):
    newMetadata = []
    columns = ["filename", "label", "class"]
    print()
    print("Writing Data...")
    for c in tqdm(imagePaths):
        class_images = imagePaths[c]
        for path in class_images:
            filename = os.path.basename(path)

            # Copy image to new processed images path
            os.makedirs(os.path.join(output_path, 'images'), exist_ok=True)
            newPath = os.path.join(output_path, 'images', filename)
            shutil.copy(path, newPath)

            # Update metadata csv
            newMetadata.append([filename, classes[c], c])
    df = pd.DataFrame(newMetadata, columns=columns)
    df.to_csv(os.path.join(output_path, 'processed_metadata.csv'), index=False)


if __name__=='__main__':
    main()
