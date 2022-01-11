import numpy as np
import shapely
from rasterio import features


def get_reduced_coords(coords, angle_th, distance_th):
    vector_rep = np.diff(coords, axis=0)
    angle_th_rad = np.deg2rad(angle_th)
    points_removed = [0]
    while len(points_removed):
        points_removed = list()
        for i in range(len(vector_rep) - 1):
            if len(coords) - len(points_removed) == 3:
                break
            v01 = vector_rep[i]
            v12 = vector_rep[i + 1]
            d01 = np.linalg.norm(v01)
            d12 = np.linalg.norm(v12)
            if d01 < distance_th and d12 < distance_th:
                points_removed.append(i + 1)
                vector_rep[i + 1] = coords[i + 2] - coords[i]
                continue
            angle = np.arccos(np.dot(v01, v12) / (d01 * d12))
            if angle < angle_th_rad:
                points_removed.append(i + 1)
                vector_rep[i + 1] = coords[i + 2] - coords[i]
        coords = np.delete(coords, points_removed, axis=0)
        vector_rep = np.diff(coords, axis=0)
    return coords.astype(int)


def reduce_polygon(polygon, angle_th=0, distance_th=0):
    ext_poly_coords = get_reduced_coords(
        np.array(polygon.exterior.coords[:]), angle_th, distance_th
    )
    interior_coords = [
        get_reduced_coords(np.array(interior.coords[:]), angle_th, distance_th)
        for interior in polygon.interiors
    ]
    return shapely.geometry.Polygon(ext_poly_coords, interior_coords)


def mask_to_polygons_layer(mask):
    all_polygons = []
    for shape, _ in features.shapes(mask.astype(np.int16), mask=(mask > 0)):
        all_polygons.append(
            reduce_polygon(shapely.geometry.shape(shape), angle_th=2, distance_th=3)
        )

    all_polygons = shapely.geometry.MultiPolygon(all_polygons)
    if not all_polygons.is_valid:
        all_polygons = all_polygons.buffer(0)
        # Sometimes buffer() converts a simple Multipolygon to just a Polygon,
        # need to keep it a Multi throughout
        if all_polygons.type == "Polygon":
            all_polygons = shapely.geometry.MultiPolygon([all_polygons])
    return all_polygons
