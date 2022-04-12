from copy import deepcopy

class Payload():
    def __init__(self):
        self.x = 0
        self.y = 0
        self.z = 0
        self.r = 0
        self.type = "None"
        self.distance = 100001
        self.bounds = None
        self.contour = None
        self.selected = 0
        self.box = None
        self.pix_centroid = [0, 0]

    def tags(self):
        tag_set = deepcopy(self.__dict__)
        tag_set.pop('pix_centroid')
        return tag_set
