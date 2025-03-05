class Geometry:
    def __init__(self):
        self.shapes = []

    def add_point(self, x, y, id):
        point = {'type': 'point', 'coordinates': (x, y), 'id': id}
        self.shapes.append(point)

    def add_line(self, start_point_id, end_point_id, id):
        line = {'type': 'line', 'start': start_point_id, 'end': end_point_id, 'id': id}
        self.shapes.append(line)

    def add_surface(self, line_ids):
        surface = {'type': 'surface', 'lines': line_ids}
        self.shapes.append(surface)

    def get_shapes(self):
        return self.shapes

    def clear_shapes(self):
        self.shapes = []