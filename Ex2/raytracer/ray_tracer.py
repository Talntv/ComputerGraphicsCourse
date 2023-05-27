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
    w = (camera.look_at - camera.position) / np.linalg.norm(camera.look_at - camera.position)
    u = np.cross(camera.up_vector, w) / np.linalg.norm(np.cross(camera.up_vector, w))
    v = np.cross(w, u)

    # Step 2: Convert pixel coordinates to NDC
    aspect_ratio = camera.screen_width / camera.screen_distance
    x_ndc = (j - camera.screen_width / 2) / (camera.screen_width / 2)
    y_ndc = (i - camera.screen_width / 2) / (camera.screen_width / 2) * aspect_ratio

    # Step 3: Calculate ray direction
    direction = w + u * x_ndc + v * y_ndc
    return direction / np.linalg.norm(direction)
    # center_point = camera.position + camera.screen_distance * camera.look_at
    # v_right = normalize(np.cross(camera.look_at, camera.up_vector))
    # v_up = normalize(np.cross(v_right, camera.look_at))
    # ratio = 1 # come back to this
    # p = center_point + (i - 250) * ratio * v_right - (j - 250) * ratio * v_up
    # return p

def intersection(ray, settings, objects, position):
    min_inter = 0
    # cube -
    # self.position = position
    # self.scale = scale
    # self.material_index = material_index
    def sphere_intersection(ray, object, position):
        a = 1
        b = np.dot(2*ray, position - object.position)
        c = np.linalg.norm(position-object.position)**2 - object.radius**2
        coeffs = [a, b, c]
        roots = np.roots(coeffs)
        positive_instances = roots[roots > 0]
        if len(positive_instances) == 0:
            return None
        # Find the minimal positive instance
        sphere_intersections = np.min(positive_instances)
        return sphere_intersections
    
    def infinite_plane_intersection(ray, object, position):
        V = ray - position
        dot_product = np.dot(object.normal, V)
        if abs(dot_product) >= 1e-6:
            t = (object.offset - np.dot(object.normal, position))/dot_product
            infinite_plane_intersections = position + t * V
            return min(infinite_plane_intersections) 
           
    def cube_intersection(ray, object, position):
        return
    
    for object in objects:
        if (type(object) == Sphere):
            res = sphere_intersection(ray, object, position)
            if res:
                if min_inter == 0 or res < min_inter:
                    min_inter = res
        # elif (type(object) == Cube):
        #     intersections.append(cube_intersection(ray, object, position))
        # elif (type(object) == InfinitePlane):
        #     intersections.append(infinite_plane_intersection(ray, object, position))
    ret = min_inter if min_inter > 0 else None
    return ret

def get_color(hit):
    return 255

def get_scene(camera, settings, objects, width, height):
    img = np.zeros((500, 500, 3))
    for i in range(500):
        print(i)
        for j in range(500):
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

    image_array = get_scene(camera, scene_settings, objects, 500, 500)

    # Save the output image
    save_image(image_array, args.output_image)


if __name__ == '__main__':
    main()
