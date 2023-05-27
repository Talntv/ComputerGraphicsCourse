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

def construct_ray_through_pixel(camera, i, j):
    # self.position = position  === P0
    # self.look_at = look_at === Vt0 (need to normalize)
    # self.up_vector = up_vector === Vup (need to normalize)
    # self.screen_distance = screen_distance === d
    # self.screen_width = screen_width === w
    # we want the intersection point P and the intersection direction V
    normalized_look_at = normalize(camera.look_at)
    center_point = camera.position + camera.screen_distance * normalized_look_at
    v_right = normalize(np.cross(normalized_look_at, camera.up_vector))
    v_up = normalize(np.cross(v_right, normalized_look_at))
    ratio = 1/50 # come back to this
    p = center_point + (j - 25) * ratio * v_right - (i - 25) * ratio * v_up
    return p

def intersection(pixel, settings, objects, position):
    normalized_ray = normalize(pixel - position)
    def sphere_intersection(ray, object, position):
        b = np.dot(2*ray, position - object.position)
        c = np.linalg.norm(position-object.position)**2 - object.radius**2
        delta = b**2 - 4*c
        if delta > 0:
            t1 = (-b + np.sqrt(delta)) / 2
            t2 = (-b - np.sqrt(delta)) / 2
            if t1 > 0 and t2 > 0:
                return min(t1, t2)
        return None
    
    def infinite_plane_intersection(ray, object, position):
        dot_product = np.dot(object.normal, ray)
        # If false, the ray and the plane are parallel
        if abs(dot_product) >= 1e-6:
            t = -(np.dot(object.normal, position) + object.offset) / dot_product
            infinite_plane_intersections = position + t*ray
            return infinite_plane_intersections
           
    def cube_intersection(ray, object, position):
        min_bound = object.position - object.scale / 2
        max_bound = object.position + object.scale / 2
        # Calculate the inverse direction components to avoid division in each iteration
        inv_direction = 1.0 / ray
        # Calculate the intersection intervals for each axis
        tmin = (min_bound - position) * inv_direction
        tmax = (max_bound - position) * inv_direction
        # Find the maximum and minimum intersection intervals
        tmin = np.maximum(tmin, np.min(tmax, axis=0))
        tmax = np.minimum(tmax, np.max(tmin, axis=0))
        # Check if there is a valid intersection
        if np.any(tmax < 0) or np.any(tmin > tmax):
            return None
        # Calculate the intersection point
        t = np.max(tmin)
        intersection_point = position + ray * t

        return intersection_point
    
    distances = []
    for object in objects:
        if (type(object) == Sphere):
            distance = sphere_intersection(normalized_ray, object, position)
            if distance:
                distances.append(distance)
        elif (type(object) == Cube):
            distance = cube_intersection(normalized_ray, object, position)
            if distance:
                distances.append(distance)
        elif (type(object) == InfinitePlane):
            distance = infinite_plane_intersection(normalized_ray, object, position)
            if distance:
                distances.append(distance)
    return min(distances)

def get_color(hit):
    return 255

def get_scene(camera, settings, objects, width, height):
    img = np.zeros((50, 50, 3))
    for i in range(50):
        print(i)
        for j in range(50):
            # Get the direction (V) and intersection point (P) with the image plane
            ray = construct_ray_through_pixel(camera, i, j)
            hit = intersection(ray, settings, objects, camera.position)
            if hit:
                img[i][j] = get_color(hit)
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

    image_array = get_scene(camera, scene_settings, objects, 400, 400)

    # Save the output image
    save_image(image_array, args.output_image)


if __name__ == '__main__':
    main()
