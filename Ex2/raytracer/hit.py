from surfaces.infinite_plane import InfinitePlane
import numpy as np

def normalize(vector):
    magnitude = np.linalg.norm(vector)
    if magnitude == 0:
        return vector
    return vector / magnitude

class Hit:
    def __init__(self, t, surface, point):
        self.t = t
        self.surface = surface
        self.point = point
        self.normal = normalize(self.point - surface.position) if type(surface) != InfinitePlane else normalize(surface.normal)