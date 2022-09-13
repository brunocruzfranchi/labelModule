import json
from datetime import datetime
from traceback import print_tb


class COCO:
    def __init__(self, set_HIBA=True):
        self._info = {}
        self._licenses = []
        self._images = []
        self._annotations = []
        self._categories = []

        if set_HIBA: self.addLicense(License(1, 'Hospital Italiano', 'https://www.hospitalitaliano.org.ar'))

    def setInfo(self, info):
        self._info = info

    def addLicense(self, _license):
        self._licenses.append(_license)

    def addImage(self, image):
        self._images.append(image)

    def addAnnotation(self, annotation):
        self._annotations.append(annotation)

    def addCategories(self, categories):
        self._categories.append(categories)

    def getLicense(self):
        return self._licenses

    def toJson(self):
        annotations = []
        categories = []
        licenses = []
        images = []

        # Convert images to JSON
        for image in self._images:
            images.append(image.toJson())

        # Convert licenses to Json
        for license in self._licenses:
            licenses.append(license.toJson())

        # Convert annotations
        for annotation in self._annotations:
            annotations.append(annotation.toJson())

        # Convert categories
        for category in self._categories:
            categories.append(category.toJson())

        return {
            'info': self._info.toJson(),
            'licenses': licenses,
            'images': images,
            'annotations': annotations,
            'categories': categories
        }

    def exportJson(self, outputPath, fileName):
        # os.getcwd() Obtains actual workplace path
        with open(outputPath + fileName, 'w', encoding='utf-8') as f:
            json.dump(self.toJson(), f, ensure_ascii=False, indent=4)


# info{ "year": int, "version": str, "description": str, "contributor": str, "url": str, "date_created": datetime, }
class Info:
    def __init__(self, version: str = "1.0.0", description: str = "", contributor: str = "", url: str = ""):
        self._version = version
        self._description = description
        self._contributor = contributor
        self._url = url
        self._year = int(datetime.now().strftime('%Y'))
        self._date_created = datetime.now().strftime('%Y-%m-%d')

    def toJson(self):
        return {
            'year': self._year,
            'version': self._version,
            'description': self._description,
            'contributor': self._contributor,
            'url': self._url,
            'date_created': self._date_created
        }


# license{ "id": int, "name": str, "url": str, }
class License:
    def __init__(self, license_id: int, name: str, url: str):
        self.id = license_id
        self.name = name
        self.url = url

    def getId(self):
        return self.id

    def toJson(self):
        return {
            'id': self.id,
            'name': self.name,
            'url': self.url
        }


# categories[{ "id": int, "name": str, "supercategory": str, "keypoints": [str], "skeleton": [edge],}]
class Categories:
    def __init__(self, category_id: int, name: str, supercategory: str, keypoints, skeleton):
        self.category_id = category_id
        self.name = name
        self.supercategory = supercategory
        self.keypoints = keypoints
        self.skeleton = skeleton

    def toJson(self):
        return {
            'id': self.category_id,
            'name': self.name,
            'supercategory': self.supercategory,
            'keypoints': self.keypoints,
            'skeleton': self.skeleton
        }


# image{ "id": int, "width": int, "height": int, "file_name": str, "license": int, "flickr_url": str, "coco_url": str, "date_captured": datetime, }
class ImageCOCO:
    def __init__(self, image_id: int, width: int, height: int, file_name: str, license_id: int,
                 date_captured=None, flick_url: str = '', coco_url: str = ''):
        self.id = image_id
        self.width = width
        self.height = height
        self.file_name = file_name
        self.license = license_id
        self.flickr_url = flick_url
        self.coco_url = coco_url
        if not date_captured:
            date_captured = datetime.now().strftime('%Y-%m-%d')
        self.date_captured = date_captured

    def getId(self) -> int:
        return self.id

    def toJson(self) -> dict:
        return {
            'id': self.id,
            'width': self.width,
            'height': self.height,
            'file_name': self.file_name,
            'license': self.license,
            'flickr_url': self.flickr_url,
            'coco_url': self.coco_url,
            'date_captured': self.date_captured
        }


class Annotation:
    def __init__(self, annotation_id: int, image_id: int, category_id: int, segmentation: list = None,
                 area: float = None, bbox: list = None, iscrowd: int = 0,
                 keypoints: list = None, num_keypoints: int = None):

        self.id = annotation_id
        self.image_id = image_id
        self.category_id = category_id
        self.segmentation = segmentation
        self.area = area
        self.bbox = bbox
        self.iscrowd = iscrowd
        self.keypoints = keypoints
        self.num_keypoints = num_keypoints

    def toJson(self) -> dict:
        json_dict = {}
        for var in self.__dict__:
            if getattr(self, var) is not None:
                json_dict[var] = getattr(self, var)
        return json_dict
