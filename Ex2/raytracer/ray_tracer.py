import argparse
from PIL import Image
import numpy as np

from camera import Camera
from light import Light
from material import Material
from scene_settings import SceneSettings
from surfaces.cube import Cube
from surfaces.infinite_plane import InfinitePlane
from surfaces.sphere import Sphere

def normalize(vector):
    magnitude = np.linalg.norm(vector)
    if magnitude == 0:
        return vector
    return vector / magnitude

def normalize_multiple(vectors):
    normalized_vectors = np.copy(vectors)
    for i in range(len(normalized_vectors)):
        for j in range(len(normalized_vectors[0])):
            normalized_vectors[i][j] = normalize(vectors[i][j])
    return normalized_vectors

def get_normalized_camera_parameters(camera, width):
    normalized_look_at = normalize(camera.look_at)
    center_point = camera.position + camera.screen_distance * normalized_look_at
    v_right = normalize(np.cross(normalized_look_at, camera.up_vector))
    v_up = normalize(np.cross(v_right, normalized_look_at))
    ratio = 1/width
    return (center_point, v_up, v_right, ratio)

def get_rays(center_point, v_up, v_right, ratio, i_coords, j_coords, width, height):
    v_up = v_up.reshape(1, 1, 3) 
    v_right = v_right.reshape(1, 1, 3)
    return center_point + np.expand_dims((j_coords - width//2), 2) * ratio * v_right - np.expand_dims((i_coords - height//2), 2) * ratio * v_up

def intersections(rays, objects, camera_origin):
    hits = []
    min_hits = np.empty(rays.shape[:2], dtype=np.ndarray)

    for object in objects:
        if isinstance(object, Sphere):
            oc = camera_origin - object.position
            b = 2* np.sum(rays * oc, axis=-1)
            c = np.sum(oc**2, axis=-1) - object.radius**2
            delta = b**2 - 4 * c
            t1 = (-b + np.sqrt(np.maximum(delta, 0))) / 2
            t2 = (-b - np.sqrt(np.maximum(delta, 0))) / 2
            t = np.where(delta >= 0, np.minimum(t1, t2), np.inf)
            hits.append((t, object))

        elif isinstance(object, InfinitePlane):
            dot_product = np.sum(object.normal * rays, axis=-1)  # Dot product calculation
            valid_hits = np.abs(dot_product) >= 1e-6
            t = np.where(valid_hits, -(np.dot(object.normal, camera_origin) + object.offset) / dot_product, np.inf)
            hits.append((t, object))

        elif isinstance(object, Cube):
            t_min = np.full_like(rays, -np.inf)
            t_max = np.full_like(rays, np.inf)
            for i in range(3):
                pos_face = object.position[i] + (object.scale / 2)
                neg_face = object.position[i] - (object.scale / 2)
                mask = np.abs(rays[..., i]) > 0
                t1 = np.where(mask, (neg_face - camera_origin[..., i]) / rays[..., i], np.inf)
                t2 = np.where(mask, (pos_face - camera_origin[..., i]) / rays[..., i], -np.inf)
                t_min[..., i] = np.minimum(t1, t2)
                t_max[..., i] = np.maximum(t1, t2)
            t_enter = np.amax(t_min, axis=-1)
            t_exit = np.amin(t_max, axis=-1)
            hits.append((np.where(t_enter <= t_exit, t_enter, np.inf), object))

    # Combine all hits into a single array
    if hits:
        for i in range(rays.shape[0]):
            for j in range(rays.shape[1]):
                min_hit = np.Inf
                min_obj = ""
                for q in range(len(objects)):
                    if hits[q][0][i][j] <= min_hit:
                        min_hit = hits[q][0][i][j]
                        min_obj = hits[q][1]
                min_hits[i][j] = [min_hit,min_obj]
        return min_hits

    return None    

def get_hits(rays, objects, position):
    normalized_rays = normalize_multiple(rays-position) 
    return intersections(normalized_rays, objects, position)
    # for object in objects:
    #     distance = intersections(rays, object, position, type(object))
    #     if distance:
    #             distances.append([distance, object])
    # min_array = min(distances, key=lambda x: x[0]) if distances else None
    # return min_array

def get_color(hit, materials):
    if not hit:
        return 0 
    idx = hit[1].material_index
    return 255*np.asarray(materials[idx-1].diffuse_color)

def split_objects(objects):
    materials = []
    surfaces = []
    lights = []
    for object in objects:
        if type(object) is Material:
            materials.append(object)
        elif type(object) is Light:
            lights.append(object)
        else:
            surfaces.append(object)
    return (materials, surfaces, lights)

def get_scene(camera, settings, objects, width, height):
    img = np.zeros((height, width, 3))
    materials, surfaces, lights = split_objects(objects)
    center_point, v_up, v_right, ratio = get_normalized_camera_parameters(camera, width)
    i_coords, j_coords = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    rays = get_rays(center_point, v_up, v_right, ratio, i_coords, j_coords, width, height)
    hits = get_hits(rays, surfaces, camera.position)
    for i in range(height):
        for j in range(width):
            img[i][j] = get_color(hits[i][j], materials)
    return img

def parse_scene_file(file_path):
    objects = []
    camera = None
    scene_settings = None
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            obj_type = parts[0]
            params = [float(p) for p in parts[1:]]
            if obj_type == "cam":
                camera = Camera(np.asarray(params[:3]), np.asarray(params[3:6]), np.asarray(params[6:9]), params[9], params[10])
            elif obj_type == "set":
                scene_settings = SceneSettings(params[:3], params[3], params[4])
            elif obj_type == "mtl":
                material = Material(params[:3], params[3:6], params[6:9], params[9], params[10])
                objects.append(material)
            elif obj_type == "sph":
                sphere = Sphere(np.asarray(params[:3]), params[3], int(params[4]))
                objects.append(sphere)
            elif obj_type == "pln":
                plane = InfinitePlane(np.asarray(params[:3]), params[3], int(params[4]))
                objects.append(plane)
            elif obj_type == "box":
                cube = Cube(np.asarray(params[:3]), params[3], int(params[4]))
                objects.append(cube)
            elif obj_type == "lgt":
                light = Light(params[:3], params[3:6], params[6], params[7], params[8])
                objects.append(light)
            else:
                raise ValueError("Unknown object type: {}".format(obj_type))
    return camera, scene_settings, objects


def save_image(image_array, new_image_path):
    image = Image.fromarray(np.uint8(image_array))
    # Save the image to a file
    image.save(new_image_path)

def main():
    parser = argparse.ArgumentParser(description='Python Ray Tracer')
    parser.add_argument('scene_file', type=str, default=".\scenes\pool.txt", help='Path to the scene file')
    parser.add_argument('output_image', type=str, default="dummy_output.png", help='Name of the output image file')
    parser.add_argument('--width', type=int, default=500, help='Image width')
    parser.add_argument('--height', type=int, default=500, help='Image height')
    args = parser.parse_args()

    # Parse the scene file
    camera, scene_settings, objects = parse_scene_file(args.scene_file)

    image_array = get_scene(camera, scene_settings, objects, args.width, args.height)

    # Save the output image
    save_image(image_array, args.output_image)


if __name__ == '__main__':
    main()
