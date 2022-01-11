from cytomine import Cytomine
from cytomine.models import (
    Annotation,
    AnnotationCollection,
    AnnotationTerm,
    TermCollection,
    ImageInstanceCollection,
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
            Annotation(
                location=location,
                id_image=id_image,
                id_project=id_project,
            )
        )
    return annotations


def upload_annotations_with_terms(annotations, id_term):
    try:
        results = annotations.save()
    except CollectionPartialUploadException as e:
        print(e)
    for _, (_, message) in results:
        ids = map(int, message.split()[1].split(","))
        for id in ids:
            AnnotationTerm(id_annotation=id, id_term=id_term).save()


def upload_to_cytomine(
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

        upload_annotations_with_terms(annotations, id_term)
