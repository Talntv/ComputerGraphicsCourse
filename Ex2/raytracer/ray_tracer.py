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

# Should only be called once, when initializing the scene
def get_normalized_camera_parameters(camera, width):
    normalized_look_at = normalize(camera.look_at - camera.position)
    center_point = camera.position + camera.screen_distance * normalized_look_at
    v_right = normalize(np.cross(camera.up_vector, normalized_look_at))
    v_up = normalize(np.cross(normalized_look_at, v_right))
    ratio = camera.screen_width / width
    return (center_point, v_up, v_right, ratio)

def get_rays(center_point, v_up, v_right, ratio, i_coords, j_coords, width, height):
    '''
    Returns ndarray of shape (height, weight, 3), where each entry in the intersection point of the ray originated in the camera position with the screen, i.e 'P'.
    '''
    v_up = v_up.reshape(1, 1, 3) 
    v_right = v_right.reshape(1, 1, 3)
    return center_point + np.expand_dims((j_coords - width//2), 2) * ratio * v_right - np.expand_dims((i_coords - height//2), 2) * ratio * v_up

def intersections(rays, surfaces, camera_origin):
    hits = []
    rays_hits = np.empty(rays.shape[:2], dtype=np.ndarray)
    light_hits = []
    for surface in surfaces:
        if isinstance(surface, Sphere):
            oc = camera_origin - surface.position
            b = 2* np.sum(rays * oc, axis=-1)
            c = np.sum(oc**2, axis=-1) - surface.radius**2
            delta = b**2 - 4 * c
            t1 = (-b + np.sqrt(np.maximum(delta, 0))) / 2
            t2 = (-b - np.sqrt(np.maximum(delta, 0))) / 2
            t = np.where(delta >= 0, np.minimum(t1, t2), np.inf)
            hits.append((t, surface))

        elif isinstance(surface, InfinitePlane):
            dot_product = np.sum(surface.normal * rays, axis=-1)
            # Exclude parallel rays to the plane
            valid_hits = np.abs(dot_product) >= 1e-6
            t = np.where(valid_hits, (np.dot(surface.offset - camera_origin, surface.normal)) / dot_product, np.inf)
            hits.append((t, surface))

        elif isinstance(surface, Cube):
            t_min = np.full_like(rays, -np.inf)
            t_max = np.full_like(rays, np.inf)
            for i in range(3):
                pos_face = surface.position[i] + (surface.scale / 2)
                neg_face = surface.position[i] - (surface.scale / 2)
                mask = np.abs(rays[..., i]) >= 0
                t1 = np.where(mask, (neg_face - camera_origin[..., i]) / rays[..., i], np.inf)
                t2 = np.where(mask, (pos_face - camera_origin[..., i]) / rays[..., i], -np.inf)
                t_min[..., i] = np.minimum(t1, t2)
                t_max[..., i] = np.maximum(t1, t2)
            t_enter = np.amax(t_min, axis=-1)
            t_exit = np.amin(t_max, axis=-1)
            hits.append((np.where(t_enter <= t_exit, t_enter, np.inf), surface))

    # Combine all hits into a single array
    if hits and rays.ndim > 1:
        for i in range(rays.shape[0]):
            for j in range(rays.shape[1]):
                for q in range(len(surfaces)):
                    if hits[q][0][i][j] != np.inf:
                        if rays_hits[i][j] is None:
                            rays_hits[i][j] = [[hits[q][0][i][j], hits[q][1]]]
                        else:
                            rays_hits[i][j].append([hits[q][0][i][j], hits[q][1]])
                rays_hits[i][j] = sorted(rays_hits[i][j], key=lambda x: x[0])
        return rays_hits
    
    elif hits:
        for q in range(len(surfaces)):
            if hits[q][0] != np.inf:
                light_hits.append([hits[q][0], hits[q][1]])
        return light_hits
    return None    

def get_hits(rays, surfaces, position):
    # Sine the rays are actually the intersection points with the screen, we convert them to normalized direction vectors in direction from the origin (camera.position) to the screen pixel
    normalized_rays = normalize_multiple(rays-position) 
    return intersections(normalized_rays, surfaces, position)

def get_light_intensity_at_point(intersection_point, hit, light, surfaces, settings):
    num_of_shadow_rays = int(settings.root_number_shadow_rays)
    direction_from_light = normalize(intersection_point - light.position)
    if np.argmax(np.abs(direction_from_light)) != 0:
        right_vector = normalize(np.cross(direction_from_light, np.asarray([1, 0, 0])))
    else:
        right_vector = normalize(np.cross(direction_from_light, np.asarray([0, 1, 0])))
    up_vector = normalize(np.cross(direction_from_light, right_vector))
    successful_light_rays_hits = 0
    cell_size = light.radius / num_of_shadow_rays
    for i in range(num_of_shadow_rays):
        for j in range(num_of_shadow_rays):
            p = light.position + (j - num_of_shadow_rays // 2) * cell_size * right_vector - (i - num_of_shadow_rays // 2) * cell_size * up_vector
            right = np.random.uniform(-0.5 * cell_size, 0.5 * cell_size) * right_vector
            up = np.random.uniform(-0.5 * cell_size, 0.5 * cell_size) * up_vector
            p = p + right + up
            ray_to_light_direction = normalize(intersection_point - p)
            light_hit = sorted(intersections(ray_to_light_direction, surfaces, p), key=lambda x: x[0])[0][1]
            # i.e ray hit the surface in the original intersection point
            if light_hit == hit[0][1]:
                successful_light_rays_hits += 1
    # Formula from bottom of page 6
    return (1 - light.shadow_intensity) + light.shadow_intensity * (successful_light_rays_hits / (num_of_shadow_rays**2))

def get_color(hit, ray, bg, materials, lights, surfaces, camera, settings, recursion_depth):
    color = np.zeros((3))
    V = normalize(ray - camera.position)
    if not hit or recursion_depth > 3:
        return bg 
    material = materials[hit[0][1].material_index-1]
    intersection_point = camera.position + hit[0][0] * V
    normal_to_surface = normalize(intersection_point - hit[0][1].position) if type(hit[0][1]) != InfinitePlane else normalize(hit[0][1].normal)
    for light in lights:
        direction_to_light = normalize(light.position - intersection_point)
        # To produce soft shadows
        light_intensity = get_light_intensity_at_point(intersection_point, hit, light, surfaces, settings)
        # Kd(N.dot(Li))
        color += material.diffuse_color * np.maximum(normal_to_surface.dot(direction_to_light), 0) * light.color * light_intensity
        # Ks((V.dot(R))^n)
        direction_to_ray_origin = normalize(camera.position - intersection_point)
        phong = np.clip(normal_to_surface.dot(normalize((direction_to_light + direction_to_ray_origin))), 0, 1)
        color += np.power(phong, material.shininess) * material.specular_color * light.specular_intensity * light.color * light_intensity
    return 255 * np.clip(color, 0, 1)

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
            img[i][j] = get_color(hits[i][j], rays[i][j], settings.background_color, materials, lights, surfaces, camera, settings, 1)
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
                scene_settings = SceneSettings(np.asarray(params[:3]), params[3], params[4])
            elif obj_type == "mtl":
                material = Material(np.asarray(params[:3]), np.asarray(params[3:6]), np.asarray(params[6:9]), params[9], params[10])
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
                light = Light(np.asarray(params[:3]), np.asarray(params[3:6]), params[6], params[7], params[8])
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
    parser.add_argument('--width', type=int, default=100, help='Image width')
    parser.add_argument('--height', type=int, default=100, help='Image height')
    args = parser.parse_args()

    # Parse the scene file
    camera, scene_settings, objects = parse_scene_file(args.scene_file)
    np.seterr(divide='ignore', invalid='ignore')

    image_array = get_scene(camera, scene_settings, objects, args.width, args.height)

    # Save the output image
    save_image(image_array, args.output_image)


if __name__ == '__main__':
    main()
