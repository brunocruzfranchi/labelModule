import os
import numpy as np
import pandas as pd
from src.data.coco_class import *


def get_keypoints_fromDF(df_values: pd.DataFrame, width: int, height: int):
    x = ((df_values['value.x'].to_numpy() / 100.) * width).astype(int)
    y = ((df_values['value.y'].to_numpy() / 100.) * height).astype(int)

    # reshape to [length,] to [length, 1]
    x = x.reshape((x.shape[0], 1))
    y = y.reshape((y.shape[0], 1))
    visibility = np.ones((x.shape[0], 1)) * 2

    # return numpy array (x.shape[0],3) -> (x, y, visibility)
    return np.concatenate((x, y, visibility), axis=1, dtype='float64')


def get_labels_fromDF(df_values: pd.DataFrame):
    labels = [label[0] for label in df_values['value.keypointlabels']]
    return labels


def get_annotations(annotation_path: str, file_path: str):
    annotations_json = json.load(open(os.path.join(annotation_path, file_path)))
    return annotations_json
    
def get_annotation_information(anno_json):
    # Extract Name from annotation:
    fileName = (pd.json_normalize(data=anno_json)['file_upload'][0]).split('-')[-1]

    # Extract ID (Accession Number) from annotation:
    fileID = fileName.split('_')[2]

    # Dataframe Values
    df_values = pd.json_normalize(data=anno_json, record_path=['annotations', ['result']])

    # Obtain size of the image
    width = df_values['original_width'].unique()[0]
    height = df_values['original_height'].unique()[0]

    # Get all keypoint and transform them to [x, y, visibility = 2]
    keypoints: np.ndarray = get_keypoints_fromDF(df_values, width, height)

    return int(fileID), fileName, int(width), int(height), keypoints


def create_LabelStudioDataset(root_anno_path, annotation_files):
    coco_ds = COCO(set_HIBA=True)
    coco_ds.setInfo(Info(version="1.0.0", description="Spinogram_Dataset"))
    coco_ds.addCategories(Categories(category_id=1, name='spine', supercategory='spine',
                                     keypoints=['C2OT', 'C1AE', 'C1PE', 'C2CE', 'C2AI', 'C2PI', 'C7AS', 'C7PS', 'C7CE',
                                                'C7AI', 'C7PI', 'T1AS', 'T1PS', 'T1CE', 'T1AI', 'T1PI', 'T5AS', 'T5PS',
                                                'T12AI', 'T12PI', 'L1AS', 'L1PS', 'L4AS', 'L4PS', 'L4AI', 'L4PI', 'S1AS',
                                                'S1MI', 'S1PS', 'F1HC', 'F2HC'],
                                     skeleton=[]))

    annotationNumber = 1

    # ToDo: make a new function that gets all annotations from all json files, and then
    # we can use a simple for loop to get create cocoDataset

    for file_path in annotation_files:
        
        annotations = get_annotations(root_anno_path, file_path)

        for json_anno in annotations:

            fileId, fileName, width, height, keypoints = get_annotation_information(json_anno)

            cocoImage = ImageCOCO(image_id=fileId, width=width, height=height,
                                file_name=fileName, license_id=coco_ds.getLicense()[0].getId())

            coco_ds.addImage(cocoImage)

            # Create annotation for image
            annotation = Annotation(annotation_id=annotationNumber, image_id=cocoImage.getId(),
                                    category_id=1, keypoints=[coordinate for row in keypoints for coordinate in row],
                                    num_keypoints=keypoints.shape[0])

            coco_ds.addAnnotation(annotation)

            annotationNumber += 1

    return coco_ds
