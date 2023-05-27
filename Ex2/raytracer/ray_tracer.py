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

def get_normalized_camera_parameters(camera, width):
    normalized_look_at = normalize(camera.look_at)
    center_point = camera.position + camera.screen_distance * normalized_look_at
    v_right = normalize(np.cross(normalized_look_at, camera.up_vector))
    v_up = normalize(np.cross(v_right, normalized_look_at))
    ratio = 1/width
    return (center_point, v_up, v_right, ratio)

def get_ray(center_point, v_up, v_right, ratio, i, j, width, height):
    return center_point + (j - width//2) * ratio * v_right - (i - height//2) * ratio * v_up

def intersections(ray, object, camera_origin, intersection_type):
    if intersection_type is Sphere:
        oc = camera_origin - object.position
        b = 2*np.dot(ray, oc)
        c = np.dot(oc, oc) - object.radius**2
        delta = b**2 - 4*c
        # -> 2 intersected points
        if delta > 0:
            t1 = (-b + np.sqrt(delta)) / 2
            t2 = (-b - np.sqrt(delta)) / 2
            if t1 > 0 and t2 > 0:
                return min(t1, t2)
        return None
    
    elif intersection_type is InfinitePlane:
        dot_product = np.dot(object.normal, ray)
        # If false, the ray and the plane are parallel
        if abs(dot_product) >= 1e-6:
            t = -(np.dot(object.normal, camera_origin) + object.offset) / dot_product
            return t if t > 0 else None
           
    elif intersection_type is Cube:
        t_min = np.full_like(camera_origin, -np.inf)
        t_max = np.full_like(camera_origin, np.inf)
        # x, y, z axes
        for i in range(3):
            if abs(ray[i]) > 0:
                # find the intersections with 2 parallel faces in the same axis
                t1 = (object.position[i] - (object.scale / 2) - camera_origin[i]) / ray[i]
                t2 = (object.position[i] + (object.scale / 2) - camera_origin[i]) / ray[i]
                t_min[i] = min(t1, t2)
                t_max[i] = max(t1, t2)
        t_enter = np.max(t_min)
        t_exit = np.min(t_max)
        # If there exists some face of the cube that was hit before all 3 faces were hit,
        # The ray doesn't hit the cube.
        return t_enter if t_enter <= t_exit else None
    
    else:
        return

def get_hit(pixel, objects, position):
    normalized_ray = normalize(pixel - position)
    distances = []
    for object in objects:
        distance = intersections(normalized_ray, object, position, type(object))
        if distance:
                distances.append([distance, object])
    min_array = min(distances, key=lambda x: x[0]) if distances else None
    return min_array

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
    for i in range(height):
        print(i)
        for j in range(width):
            ray = get_ray(center_point, v_up, v_right, ratio, i, j, width, height)
            hit = get_hit(ray, surfaces, camera.position)
            img[i][j] = get_color(hit, materials)
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
