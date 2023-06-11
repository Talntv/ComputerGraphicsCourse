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
from hit import Hit

import os
from joblib import Parallel, delayed
import time
def measure_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Function {func.__name__} took {execution_time:.3f} seconds")
        return result
    return wrapper

def normalize(vectors):
    # Normalize single vector
    if vectors.ndim == 1:
        magnitude = np.linalg.norm(vectors)
        if magnitude == 0:
            return vectors
        return vectors / magnitude
    # Normalize array of vectors
    norms = np.linalg.norm(vectors, axis=2)
    normalized_vectors = vectors / norms[..., np.newaxis]
    return normalized_vectors

# Should only be called once, when initializing the scene
def get_normalized_camera_parameters(camera, width):
    normalized_look_at = normalize(camera.look_at - camera.position)
    center_point = camera.position + camera.screen_distance * normalized_look_at
    v_right = normalize(np.cross(camera.up_vector, normalized_look_at))
    v_up = normalize(np.cross(normalized_look_at, v_right))
    ratio = camera.screen_width / width
    return (center_point, v_up, v_right, ratio)

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

def get_rays(camera, center_point, v_up, v_right, ratio, i_coords, j_coords, width, height):
    '''
    Returns ndarray of shape (height, weight, 3), where each entry in the intersection point of the ray originated in the camera position with the screen, i.e 'P'.
    '''
    v_up = v_up.reshape(1, 1, 3) 
    v_right = v_right.reshape(1, 1, 3)
    screen_p =  center_point + np.expand_dims((j_coords - width//2), 2) * ratio * v_right - np.expand_dims((i_coords - height//2), 2) * ratio * v_up
    return normalize(screen_p - camera.position)

def intersections(rays, surfaces, camera_origin):
    np.seterr(divide='ignore', invalid='ignore')
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
                            rays_hits[i][j] = [Hit(hits[q][0][i][j], hits[q][1], camera_origin + hits[q][0][i][j]*rays[i][j])]
                        else:
                            rays_hits[i][j].append(Hit(hits[q][0][i][j], hits[q][1], camera_origin + hits[q][0][i][j]*rays[i][j]))
                rays_hits[i][j] = sorted(rays_hits[i][j], key=lambda x: x.t)
        return rays_hits
    
    # elif hits:
    #     for q in range(len(surfaces)):
    #         if hits[q][0] != np.inf and hits[q][0] > 0.0000001:
    #             light_hits.append(Hit(hits[q][0], hits[q][1], camera_origin + hits[q][0]*rays))
    #     return sorted(light_hits, key=lambda x: x.t)
    return None    

def get_light_intensity_at_point(hit, light, surfaces, settings):
    num_of_shadow_rays = int(settings.root_number_shadow_rays)
    direction_from_light = normalize(hit.point - light.position)
    light_normal_dimension = 1 if np.argmax(np.abs(direction_from_light)) != 0 else 0
    right_vector = normalize(np.cross(direction_from_light, np.asarray([light_normal_dimension, 1-light_normal_dimension, 0]))).reshape(1, 1, 3) 
    up_vector = normalize(np.cross(direction_from_light, right_vector)).reshape(1, 1, 3) 
    successful_light_rays_hits = 0
    cell_size = light.radius / num_of_shadow_rays
    indices = np.arange(num_of_shadow_rays) - num_of_shadow_rays // 2
    j_indices, i_indices = np.meshgrid(indices, indices)

    # Calculate positions
    p = light.position + np.expand_dims(j_indices,2) * cell_size * right_vector - np.expand_dims(i_indices,2) * cell_size * up_vector

    # Generate random vectors
    right = np.expand_dims(np.random.uniform(-0.5 * cell_size, 0.5 * cell_size, size=(num_of_shadow_rays, num_of_shadow_rays)),2) * right_vector
    up = np.expand_dims(np.random.uniform(-0.5 * cell_size, 0.5 * cell_size, size=(num_of_shadow_rays, num_of_shadow_rays)),2) * up_vector

    # Update positions
    p += right + up

    # Calculate ray directions
    ray_to_light_direction = normalize(hit.point - p)

    # Perform intersection calculations
    light_hits = intersections(ray_to_light_direction, surfaces, p)
    matching_surfaces = np.array([obj[0].surface for obj in light_hits.ravel()]) == hit.surface
    successful_light_rays_hits = np.count_nonzero(matching_surfaces)
    # for i in range(num_of_shadow_rays):
    #     for j in range(num_of_shadow_rays):
    #         p = light.position + (j - num_of_shadow_rays // 2) * cell_size * right_vector - (i - num_of_shadow_rays // 2) * cell_size * up_vector
    #         right = np.random.uniform(-0.5 * cell_size, 0.5 * cell_size) * right_vector
    #         up = np.random.uniform(-0.5 * cell_size, 0.5 * cell_size) * up_vector
    #         p += right + up
    #         ray_to_light_direction = normalize(hit.point - p)
    #         light_hit = intersections(ray_to_light_direction, surfaces, p)[0]
    #         # i.e ray hit the surface in the original intersection point
    #         if light_hit.surface == hit.surface:
    #             successful_light_rays_hits += 1
    # Formula from bottom of page 6
    return (1 - light.shadow_intensity) + light.shadow_intensity * (successful_light_rays_hits / (num_of_shadow_rays**2))

def get_reflection_color(hit : Hit, ray_origin, materials, lights, surfaces, settings, material, recursion_depth):
    ray_to_camera_direction = normalize(ray_origin - hit.point)
    ray_reflected_direction = 2 * np.dot(hit.normal, ray_to_camera_direction) * hit.normal - ray_to_camera_direction
    hits = intersections(ray_reflected_direction, surfaces, hit.point)
    if not hits:
        return settings.background_color * material.reflection_color
    return get_color(hits, hit.point, materials, lights, surfaces, settings, recursion_depth + 1) * material.reflection_color

def get_color(hits : Hit, ray_origin, materials, lights, surfaces, settings, recursion_depth):
    if not hits or recursion_depth > settings.max_recursions:
        return settings.background_color
    diffuse_color = np.zeros(3)
    specular_color = np.zeros(3)
    hit = hits[0]
    material = materials[hit.surface.material_index-1]
    for light in lights:
        direction_to_light = normalize(light.position - hit.point)
        # To produce soft shadows
        light_intensity = get_light_intensity_at_point(hit, light, surfaces, settings)
        # Kd(N.dot(Li))
        diffuse_color += material.diffuse_color * np.maximum((hit.normal).dot(direction_to_light), 0) * light.color * light_intensity
        # Ks((V.dot(R))^n)
        direction_to_ray_origin = normalize(ray_origin - hit.normal)
        phong = np.clip(hit.normal.dot(normalize((direction_to_light + direction_to_ray_origin))), 0, 1)
        specular_color += np.power(phong, material.shininess) * material.specular_color * light.specular_intensity * light.color * light_intensity
    # Handle transparency
    bg_color = 0 if material.transparency <= 0 else get_color(hit[1:], ray_origin, materials, lights, surfaces, settings, recursion_depth + 1)
    # Handle reclection
    reflection_color = get_reflection_color(hit, ray_origin, materials, lights, surfaces, settings, material, recursion_depth)
    # Sum all colors
    return np.clip((bg_color * material.transparency + (diffuse_color + specular_color) * (1 - material.transparency) + reflection_color), 0, 1)

@measure_time
def get_scene(camera, settings, objects, width, height):
    img = np.zeros((height, width, 3))
    materials, surfaces, lights = split_objects(objects)
    center_point, v_up, v_right, ratio = get_normalized_camera_parameters(camera, width)
    i_coords, j_coords = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    rays = get_rays(camera, center_point, v_up, v_right, ratio, i_coords, j_coords, width, height)
    hits = intersections(rays, surfaces, camera.position)

    results = Parallel(n_jobs=os.cpu_count())((
        delayed(get_color)(hits[i][j], camera.position, materials, lights, surfaces, settings, 1) for i in range(width) for j in range(height)))
    for index, color in enumerate(results):
        i = index // height
        j = index % height
        img[i, j] = color*255
    # for i in range(height):
    #     for j in range(width):
    #         if (j == 0):
    #             print(i)
    #         img[i][j] = get_color(hits[i][j], camera.position, materials, lights, surfaces, settings, 1)*255
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

    image_array = get_scene(camera, scene_settings, objects, args.width, args.height)

    # Save the output image
    save_image(image_array, args.output_image)


if __name__ == '__main__':
    main()
