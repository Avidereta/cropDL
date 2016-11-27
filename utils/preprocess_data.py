"""
All work related to segments processing
"""

from math import hypot
import itertools
import pandas


# TODO: correct the case of collinearity!
class Segment(object):
    """ Create a new Segment, at coordinates x1, y1 and x2, y2 """

    def __init__(self, x1=0, y1=0, x2=0, y2=0):
        """ Create a new segment at coordinates x1, y1 and x2, y2 """
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2        

    def __str__(self):
        return "({0}, {1}, {2}, {3})".format(self.x1, self.y1, self.x2, self.y2)

    def coord(self):
        return self.x1, self.y1, self.x2, self.y2

    def length(self):
        """
        Computes the segment length
        :return: float, Euclidean norm
        """
        return hypot((self.x1 - self.x2), (self.y1 - self.y2))

    def check_intersection(self, target):
        """
        checks Intersection of segment and target
        :return: bool, True if point or segment intersection, False if no intersection
        """
        segment_endpoints = []
        left = max(min( self.x1, self.x2), min(target.x1, target.x2))
        right = min(max(self.x1, self.x2), min(target.x1, target.x2))
        top = max(min(self.y1, self.y2), min(target.y1, target.y2))
        bottom = min(max(self.y1, self.y2), min(target.y1, target.y2))

        # 'NO INTERSECTION'
        if top > bottom or left > right:
            segment_endpoints = []
            return False

        # 'POINT INTERSECTION'
        elif top == bottom and left == right:
            segment_endpoints.append(left)
            segment_endpoints.append(top)
            return True

        # 'SEGMENT INTERSECTION'
        else:
            segment_endpoints.append(left)
            segment_endpoints.append(bottom)
            segment_endpoints.append(right)
            segment_endpoints.append(top)
            return True


def compute_intersections_nmb(segments):
    """
    Computes number of intersections in the list of segments
    :param segments: list of segments
    :return: int, number of intersections
    """
    nmb_of_intersections = 0
    for i, (seg1, seg2) in enumerate(itertools.combinations(segments, 2)):
            if seg1.check_intersection(seg2):
                nmb_of_intersections += 1
    return nmb_of_intersections


def count_lesions_nmb(segments_list, default_coord=(0, 0, 0, 0)):
    """
    Counts number of lesions based on the set of line segments
    :param segments_list: list of segments
    :param default_coord: set of coordinates, corresponding to the absence of segments
    :return: int, number of lesions
    """
    nmb_lesions = 0

    # empty list
    if len(segments_list) == 0:
        pass
    # list with the mark of no lesions
    elif len(segments_list) == 1:
        if segments_list[0].coord() == default_coord:
            pass
        else:
            nmb_lesions = 1
    # list with segments
    else:
        nmb_intersections = compute_intersections_nmb(segments_list)
        nmb_lesions = len(segments_list) - nmb_intersections
    assert(nmb_lesions >= 0)
    return nmb_lesions


def extract_segmnets(ann_path):
    """
    Extracts Segments from annotation file
    :param ann_path: path to .csv file with annotations
    :return: list of Segments
    """
    ann = pandas.read_csv(ann_path, sep=',', delimiter=None)
    img_segments = []
    for i in range(len(ann)):
        segment = Segment(ann["x1"][i], ann["y1"][i], ann["x2"][i], ann["y2"][i])
        img_segments.append(segment)
    return img_segments
