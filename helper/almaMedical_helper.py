import os
import json
from os import walk
from typing import List, Union
import numpy as np
from pydicom import dcmread
from collections import OrderedDict
from src.data.coco_class import *


# Funcion para obtener la direccion de las anotaciones DICOM
# Dicom ref: https://dicom.nema.org/medical/dicom/current/output/chtml/part05/sect_6.2.html
def getDicomPathFiles(root_path: str):
    dicom_files = []
    for (dirpath, dirnames, filenames) in walk(root_path):
        filenames = [ fi for fi in filenames if not fi.endswith(('.jpg', '.json'))]
        if filenames:
            for filename in filenames:
                dicom_files.append(os.path.join(dirpath, filename).replace('\\', '/'))
    return dicom_files


# Funcion para obtener los tags SOP Class UID (0008,0016), SOP Instance UID (0008,0018), y Referenced SOP Class UID (0008,1150), Referenced SOP Instance UID (0008,1155) de las
# anotaciones que se encuentran dentro de (0008,1115) Referenced Series Sequence
def getReferencedTags(dicom_path: str) -> dict:
    dicom_file = dcmread(dicom_path)
    return {
        'ClassUID': dicom_file[0x0008, 0x00016].value if ([0x0008, 0x00016] in dicom_file) else "",
        'InstanceUID': dicom_file[0x0008, 0x00018].value if ([0x0008, 0x00018] in dicom_file) else "",
        'ReferencedClassUID': dicom_file[0x0008, 0x01115][0][0x0008, 0x01140][0][0x0008, 0x01150].value if (
                    [0x0008, 0x01115] in dicom_file) else "",
        'ReferencedInstanceUID': dicom_file[0x0008, 0x01115][0][0x0008, 0x01140][0][0x0008, 0x01155].value if (
                    [0x0008, 0x01115] in dicom_file) else ""
    }


# Funcion para obtener la direccion de la imagen asociada a las anotacion (Referenced SOP Class UID y Referenced SOP Instance UID) de la dicom que contiene las anotaciones (PR-Presentation State)
def getAnnotationDicoms(files_path: list):
    annot_dicom = []
    for file_path in files_path:
        dicom_file = dcmread(file_path)
        if [0x00008, 0x00060] in dicom_file and dicom_file[0x00008, 0x00060].value == 'PR':
            annot_dicom.append(file_path)
    return annot_dicom


# Funcion para obtener la imagen en donde se hicieron las anotaciones
def findDicomImage(path_annotation_dicom, files_path: list):
    image_dicom = ''
    referencedTags = getReferencedTags(path_annotation_dicom)
    for file in files_path:
        images_tags = getReferencedTags(file)
        if images_tags['ClassUID'] == referencedTags['ReferencedClassUID'] and images_tags['InstanceUID'] == \
                referencedTags['ReferencedInstanceUID']:
            image_dicom = file
    return image_dicom.split('/')[-1]


# Funcion para obtener todos los AnchorPoints de los Diametros
def findAnchorPoints(path_annotation_dicom):
    keypoints = {}

    key_order = ["C2OT", "C1AE", "C1PE", "C2CE", "C2AI", "C2PI", "C7AS", "C7PS", "C7CE",
                 "C7AI", "C7PI", "T1AS", "T1PS", "T1CE", "T1AI", "T1PI", "T5AS", "T5PS", "T12A", "T12P",
                 "L1AS", "L1PS", "L4AS", "L4PS", "L4AI", "L4PI", "S1AS", "S1MI", "S1PS", "F1HC", "F2HC"]

    label_diameter = 'AlmaLayerANN_DIAMETER'

    annot_dicom = dcmread(path_annotation_dicom)

    for graphicLayer in annot_dicom[0x0070, 0x0001]:
        if [0x0070, 0x0068] in graphicLayer:
            if graphicLayer[0x0070, 0x0068].value[0] == label_diameter:
                if [0x0070, 0x0008] in graphicLayer:
                    keypoints[graphicLayer[0x0070, 0x0068].value[1]] = graphicLayer[0x0070, 0x0008][0][
                        0x0070, 0x0014].value

        elif [0x0070, 0x0002] in graphicLayer:
            if graphicLayer[0x0070, 0x0002].value[0] == label_diameter:
                if [0x0070, 0x0008] in graphicLayer:
                    keypoints[graphicLayer[0x0070, 0x0002].value[1]] = graphicLayer[0x0070, 0x0008][0][
                        0x0070, 0x0014].value

    new_dict = OrderedDict((k, keypoints[k]) for k in key_order if k in keypoints)
    new_dict = json.loads(json.dumps(new_dict))

    return new_dict


def get_keypoints_fromDict(dict_keypoints):
    x = []
    y = []

    for idx, coordinate in dict_keypoints.items():
        x.append(int(coordinate[0]))
        y.append(int(coordinate[1]))

    x = np.array(x)
    y = np.array(y)

    # reshape to [length,] to [length, 1]
    x = x.reshape((x.shape[0], 1))
    y = y.reshape((y.shape[0], 1))
    visibility = np.ones((x.shape[0], 1)) * 2

    # return numpy array (x.shape[0],3) -> (x, y, visibility)
    return np.concatenate((x, y, visibility), axis=1, dtype='float64')


# Generar funcion getAnnotations() en donde me retorne FileID(Accession Number), FileName (direccion de la imagen DICOM), width, height, keypoints
def getAnnotations(annotation_dicom, dicom_files):

    path_dicom_image = findDicomImage(annotation_dicom, dicom_files)
    keypoints = findAnchorPoints(annotation_dicom)

    dicom_image = dcmread(path_dicom_image)

    # Extract Name from annotation:
    fileName = path_dicom_image

    # Extract ID (Accession Number) from annotation:
    fileID = dicom_image[0x0008, 0x0050].value

    # Obtain size of the image
    width = dicom_image.pixel_array.shape[1]
    height = dicom_image.pixel_array.shape[0]

    # Get all keypoint and transform them to [x, y, visibility = 2]
    keypoints: np.ndarray = get_keypoints_fromDict(keypoints)

    return int(fileID), fileName, int(width), int(height), keypoints


def create_AlmaMedicalDataset(dicom_files):
    coco_ds = COCO(set_HIBA=True)
    coco_ds.setInfo(Info(version="1.0.0", description="Spinogram_Dataset"))
    coco_ds.addCategories(Categories(category_id=1, name='spine', supercategory='spine',
                                     keypoints=['C2OT', 'C1AE', 'C1PE', 'C2CE', 'C2AI', 'C2PI', 'C7AS', 'C7PS', 'C7CE',
                                                'C7AI', 'C7PI', 'T1AS', 'T1PS', 'T1CE', 'T1AI', 'T1PI', 'T5AS', 'T5PS',
                                                'T12A', 'T12P', 'L1AS', 'L1PS', 'L4AS', 'L4PS', 'L4AI', 'L4PI', 'S1AS',
                                                'S1MI', 'S1PS', 'F1HC', 'F2HC'],
                                     skeleton=[]))

    annotationNumber = 1

    annotation_dicoms = getAnnotationDicoms(dicom_files)

    for annotation_dicom in annotation_dicoms:

        fileId, fileName, width, height, keypoints = getAnnotations(annotation_dicom, dicom_files)

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