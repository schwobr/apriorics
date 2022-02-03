from pathlib import Path
from cytomine import Cytomine
from cytomine.models import (
    Annotation,
    AnnotationCollection,
    AnnotationTerm,
    TermCollection,
    ImageInstanceCollection,
    StorageCollection,
    Project
)
from cytomine.models.collection import CollectionPartialUploadException
import shapely


def get_image_id(slidename, id_project):
    images = ImageInstanceCollection().fetch_with_filter("project", id_project)
    images = filter(lambda x: slidename in x.filename, images)

    try:
        id_image = next(images).id
    except StopIteration:
        print(
            f"Slide {slidename} was not found in cytomine, please upload it. "
            "Stopping program..."
        )
        raise RuntimeError

    return id_image


def get_term_id(term, id_project):
    if term is not None:
        terms = TermCollection().fetch_with_filter("project", id_project)
        terms = filter(lambda x: x.name == term, terms)

        try:
            id_term = next(terms).id
        except StopIteration:
            print(
                f"Term {term} was not found on cytomine, please upload it. "
                "Resuming with no term..."
            )
            id_term = None
    else:
        id_term = None

    return id_term


def polygons_to_annotations(
    polygons, id_image, id_project, object_min_size=1000, polygon_type="polygon"
):
    annotations = AnnotationCollection()
    for polygon in polygons:
        if polygon.area < object_min_size:
            continue
        if polygon_type == "box":
            bbox = shapely.geometry.box(*polygon.bounds)
            location = bbox.wkt
        else:
            location = polygon.wkt
        annotations.append(
            Annotation(location=location, id_image=id_image, id_project=id_project,)
        )
    return annotations


def upload_annotations_with_terms(annotations, id_project, id_image, id_term=None):
    try:
        annotations.save()
    except CollectionPartialUploadException as e:
        print(e)
    if id_term is not None:
        annotations = AnnotationCollection()
        annotations.project = id_project
        annotations.image = id_image
        annotations.fetch()
        for annotation in annotations:
            if not annotation.term:
                AnnotationTerm(annotation.id, id_term).save()


def upload_polygons_to_cytomine(
    polygons,
    slidename,
    host,
    public_key,
    private_key,
    id_project,
    term=None,
    polygon_type="polygon",
    object_min_size=1000,
):
    with Cytomine(host=host, public_key=public_key, private_key=private_key) as _:
        id_image = get_image_id(slidename, id_project)
        id_term = get_term_id(term, id_project)

        annotations = polygons_to_annotations(
            polygons,
            id_image,
            id_project,
            object_min_size=object_min_size,
            polygon_type=polygon_type,
        )

        upload_annotations_with_terms(
            annotations, id_project, id_image, id_term=id_term
        )


def upload_image_to_cytomine(filepath, host, public_key, private_key, id_project):
    filepath = Path(filepath)

    with Cytomine(
        host=host,
        public_key=public_key,
        private_key=private_key,
    ) as cytomine:

        # Check that the file exists on your file system
        if not filepath.exists():
            raise ValueError("The file you want to upload does not exist")

        # Check that the given project exists
        if id_project:
            project = Project().fetch(id_project)
            if not project:
                raise ValueError("Project not found")

        # To upload the image, we need to know the ID of your Cytomine storage.
        storages = StorageCollection().fetch()
        my_storage = next(
            filter(lambda storage: storage.user == cytomine.current_user.id, storages)
        )
        if not my_storage:
            raise ValueError("Storage not found")

        uploaded_file = cytomine.upload_image(
            upload_host="http://localhost-upload",
            filename=str(filepath),
            id_storage=my_storage.id,
            id_project=id_project,
        )

        print(uploaded_file)


def get_uploaded_images(host, public_key, private_key, id_project):
    with Cytomine(host=host, public_key=public_key, private_key=private_key) as _:
        images = ImageInstanceCollection().fetch_with_filter("project", id_project)
    return [image.instanceFilename for image in images]
