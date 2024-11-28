from __future__ import annotations
from dataclasses import dataclass
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle
import matplotlib.patheffects as pe
from scipy.spatial import Delaunay, Voronoi, voronoi_plot_2d
import math
import tempfile
import subprocess

import numpy as np

bsc = 'bounded-strongly-corr'
usw = 'uncorr-similar-weights'
unc = 'uncorr'

a280 = 'a280'
berlin52 = 'berlin52'
brd14051 = 'brd14051'
dsj1000 = 'dsj1000'
eil51 = 'eil51'
eil76 = 'eil76'
fnl4461 = 'fnl4461'
lin105 = 'lin105'
pcb442 = 'pcb442'
pla33810 = 'pla33810'
pr76 = 'pr76'
pr152 = 'pr152'
pr439 = 'pr439'
st70 = 'st70'
u724 = 'u724'
usa13509 = 'usa13509'


@dataclass()
class Item:
    i: int
    profit: float
    weight: float
    node: Node


@dataclass()
class Node:
    i: int
    label: int
    x: float
    y: float
    items: [Item]

    def distance(self, other: Node) -> float:
        dx = self.x - other.x
        dy = self.y - other.y

        return math.ceil(math.sqrt(dx * dx + dy * dy))

    def profit(self):
        return sum([item.profit for item in self.items])

    def weight(self):
        return sum([item.weight for item in self.items])


@dataclass(init=False, eq=False)
class Problem:
    filename: str
    dimension: int
    number_of_items: int
    capacity_of_knapsack: int
    min_speed: float
    max_speed: float
    renting_ratio: float
    speed_coefficient: float

    nodes: [Node]
    items: [Item]

    def __init__(self, name, items_per_city, category, number):
        self.name = name
        self.items_per_city = items_per_city
        self.category = category
        self.filename = None

        self.delaunay = None
        self.weight_matrix = None

        self.dimension = None
        self.number_of_items = None
        self.capacity_of_knapsack = None
        self.min_speed = None
        self.max_speed = None
        self.renting_ratio = None
        self.points = None
        self.matrix = None

        self.nodes = []
        self.items = []

        def read_key(line, key, type_convert):
            key += ':'

            if not line.startswith(key):
                return None

            value = line[len(key):]

            if len(value) == len(line):
                return None

            return type_convert(value.strip())

        node_count = int(''.join([c for c in name if c.isdigit()])) - 1
        self.filename = f'{name}_n{items_per_city * node_count}_{category}_{number:02}'

        with open(f'/home/caios/repos/flns-ttp/instances/{self.filename}.ttp') as file:
            for line in file:
                if line.startswith('NODE_COORD_SECTION'):
                    break

                if self.dimension is None: self.dimension = read_key(line, 'DIMENSION', int)
                if self.number_of_items is None: self.number_of_items = read_key(line, 'NUMBER OF ITEMS', int)
                if self.capacity_of_knapsack is None:
                    self.capacity_of_knapsack = read_key(line, 'CAPACITY OF KNAPSACK', float)
                if self.min_speed is None: self.min_speed = read_key(line, 'MIN SPEED', float)
                if self.max_speed is None: self.max_speed = read_key(line, 'MAX SPEED', float)
                if self.renting_ratio is None: self.renting_ratio = read_key(line, 'RENTING RATIO', float)

            for line in file:
                if line.startswith('ITEMS SECTION'):
                    break

                [i, x, y] = line.split()
                self.nodes.append(Node(int(i) - 1, int(i), float(x), float(y), []))

            for line in file:
                [_, profit, weight, node] = line.split()

                node = int(node) - 1
                node = self.nodes[node]

                i = len(self.items)
                item = Item(i, float(profit), float(weight), node)
                self.items.append(item)

                node.items.append(item)

        self.speed_coefficient = (self.max_speed - self.min_speed) / self.capacity_of_knapsack

    @staticmethod
    def params_from_instance_name(instance_name: str):
        (name, items_per_city, category, number) = instance_name.split('_')

        node_count = int(''.join([c for c in name if c.isdigit()])) - 1
        items_per_city = int(items_per_city.lstrip('n'))
        items_per_city //= node_count

        number = int(number, 10)

        return name, items_per_city, category, number

    @staticmethod
    def from_instance_name(instance_name: str):
        (name, items_per_city, category, number) = Problem.params_from_instance_name(instance_name)
        return Problem(name, items_per_city, category, number)

    def solve_tsp(self):
        with tempfile.NamedTemporaryFile('w', delete=False) as tsp_file:
            tsp_filename = tsp_file.name
            par_filename = tsp_filename + '.par'
            cyc_filename = tsp_filename + '.cyc'

            tsp_file.write(
                f'NAME: {self.filename}\nTYPE: TSP\nDIMENSION: {self.dimension}\nEDGE_WEIGHT_TYPE: CEIL_2D\nNODE_COORD_SECTION')
            for node in self.nodes:
                tsp_file.write(f'\n{node.i + 1} {int(node.x)} {int(node.y)}')

        with open(par_filename, 'w') as par_file:
            par_file.write(f'PROBLEM_FILE = {tsp_filename}\nOUTPUT_TOUR_FILE = {cyc_filename}\nRUNS = 1')

        subprocess.run(["vendor/LKH-3.exe", par_filename], text=True)

        lkh_route = []

        with open(cyc_filename, 'r') as cyc_file:
            is_reading = False

            for line in cyc_file.readlines():
                if line.startswith('-1'):
                    break

                if is_reading:
                    index = int(line.strip()) - 1
                    lkh_route.append(self.nodes[index])

                elif line.startswith('TOUR_SECTION'):
                    is_reading = True

        lkh_route.append(lkh_route[0])

        return lkh_route

    def solve_asymmetric_tsp(self, matrix):
        with tempfile.NamedTemporaryFile('w', delete=False) as tsp_file:
            tsp_filename = tsp_file.name
            par_filename = tsp_filename + '.par'
            cyc_filename = tsp_filename + '.cyc'

            tsp_file.write(
                f'NAME: {self.filename}\nTYPE: ATSP\nDIMENSION: {self.dimension}\nEDGE_WEIGHT_TYPE: EXPLICIT\nEDGE_WEIGHT_FORMAT: FULL_MATRIX \nEDGE_WEIGHT_SECTION')

            for line in matrix:
                line = ' '.join([str(dist) for dist in line])
                tsp_file.write('\n')
                tsp_file.write(line)
            tsp_file.write('\nEOF')

        with open(par_filename, 'w') as par_file:
            par_file.write(f'PROBLEM_FILE = {tsp_filename}\nOUTPUT_TOUR_FILE = {cyc_filename}\nRUNS = 1')

        subprocess.run(["vendor/LKH-3.exe", par_filename], text=True)

        lkh_route = []

        with open(cyc_filename, 'r') as cyc_file:
            is_reading = False

            for line in cyc_file.readlines():
                if line.startswith('-1'):
                    break

                if is_reading:
                    index = int(line.strip()) - 1
                    lkh_route.append(self.nodes[index])

                elif line.startswith('TOUR_SECTION'):
                    is_reading = True

        lkh_route.append(lkh_route[0])

        return lkh_route

    @staticmethod
    def _interp(normalize_by, data):
        if normalize_by is None:
            smallest_size = 100
            biggest_size = 4000
        else:
            smallest_size = normalize_by[0]
            biggest_size = normalize_by[1]

        min_data, max_data = min(data), max(data)
        data_range = max_data - min_data

        size_range = biggest_size - smallest_size

        sizes = [smallest_size + size_range * (size - min_data) / data_range for size in data]

        return sizes

    def _plot_neighborhood(self, nodes, nodes_x, nodes_y):
        # print(neighborhood)
        # print(nodes)
        for center in nodes:
            print(self.get_delaunay()[center])
            neighbors = [self.nodes[n] for n in self.get_delaunay()[center]]

            neighbor = neighbors.pop()

            neighbors_x = [neighbor.x]
            neighbors_y = [neighbor.y]

            while neighbors:
                next_neighbor = min(neighbors, key=neighbor.distance)
                neighbors.remove(next_neighbor)

                neighbor = next_neighbor

                neighbors_x.append(neighbor.x)
                neighbors_y.append(neighbor.y)

            neighbors_x.append(neighbors_x[0])
            neighbors_y.append(neighbors_y[0])
            plt.plot(neighbors_x, neighbors_y, 'firebrick')
            plt.scatter([nodes_x[center]], [nodes_y[center]], facecolors='none', edgecolors='k', s=[500])

    @staticmethod
    def find_node_by_label(route, label):
        label -= 1
        for node in route:
            if node.i == label:
                return node

    def plot_subgraph(self, nodes, show_index, normalize_by, with_neighbors, show_items, figsize):
        if not figsize:
            figsize=(15, 10)
        fig, ax = plt.subplots(figsize=figsize)
        # ax.set_aspect('equal', adjustable='box')

        if nodes is None:
            nodes = self.nodes

        nodes_x = [node.x for node in nodes]
        nodes_y = [node.y for node in nodes]

        size_profits = [0] + Problem._interp(normalize_by,
                                             [sum([item.profit for item in node.items]) for node in nodes[1:]])

        size_weights = [0] + Problem._interp(normalize_by,
                                             [sum([item.weight for item in node.items]) for node in nodes[1:]])

        if show_items:
            # inc_size_x = 5
            # inc_size_y = 7
            # plt.xlim([min([n.x for n in nodes]) - inc_size_x, max([n.x for n in nodes]) + inc_size_x])
            # plt.ylim([min([n.y for n in nodes]) - inc_size_y, max([n.y for n in nodes]) + inc_size_y])

            plt.scatter(nodes_x, nodes_y, facecolors='cornflowerblue', marker=MarkerStyle('o', fillstyle='top'),
                        s=size_profits)
            plt.scatter(nodes_x, nodes_y, facecolors='salmon', marker=MarkerStyle('o', fillstyle='bottom'), s=size_weights)

            plt.scatter([nodes_x[0]], [nodes_y[0]], facecolors='darkslategray', marker='*', s=1000, zorder=10,
                        path_effects=[pe.withStroke(linewidth=6, foreground='w')])

        else:
            nodes_x2 = nodes_x
            nodes_y2 = nodes_y
            # nodes_x2 = [nodes_x2[i] for i in range(len(nodes_x)) if i != 37]
            # nodes_y2 = [nodes_y2[i] for i in range(len(nodes_x)) if i != 37]
            plt.scatter(nodes_x2, nodes_y2, facecolors='cornflowerblue', marker='o', s=1000)

        if with_neighbors:
            self._plot_neighborhood(with_neighbors, nodes_x, nodes_y)

        if show_index:
            if isinstance(show_index, list):
                if isinstance(show_index[0], bool):
                    def item_label(item):
                        label = sum([1 for i in item.node.items if show_index[i.i]])
                        if label == 1:
                            label = item.node.label
                        return label

                    show_index = [(item.node, item_label(item)) for item in
                                  self.items if show_index[item.i]]
                else:
                    show_index = [Problem.find_node_by_label(nodes, label) for label in show_index]
            else:
                show_index = nodes

            if normalize_by:
                ratio = 100 / normalize_by[0]
            else:
                ratio = 1

            font_size = 18 / ratio
            font_stroke_size = 4 / ratio

            for node in show_index:
                if isinstance(node, tuple):
                    label = node[1]
                    node = node[0]
                else:
                    label = node.i + 1

                if label:
                    plt.annotate(label, (node.x, node.y), ha='center', va='center', size=font_size,
                                 path_effects=[pe.withStroke(linewidth=font_stroke_size, foreground='w')])
                    # if show_circles:
                    #     plt.scatter([node.x], [node.y], edgecolors='mediumseagreen', marker='o', facecolor='none', s=10000, linewidths=7, linestyle='--')

        plt.axis('off')

        return fig, ax

    def plot_subgraph_ks(self, nodes, show_index, normalize_by, with_neighbors, show_items, figsize, with_items):
        if not figsize:
            figsize=(15, 10)
        fig, ax = plt.subplots(figsize=figsize)
        # ax.set_aspect('equal', adjustable='box')

        if nodes is None:
            nodes = self.nodes

        nodes_x = [node.x for node in nodes]
        nodes_y = [node.y for node in nodes]

        size_profits = [0] + Problem._interp(normalize_by,
                                             [sum([item.profit for item in node.items]) for node in nodes[1:]])

        size_weights = [0] + Problem._interp(normalize_by,
                                             [sum([item.weight for item in node.items]) for node in nodes[1:]])

        if show_items:
            inc_size_x = 3
            inc_size_y = 5
            plt.xlim([min([n.x for n in nodes]) - inc_size_x, max([n.x for n in nodes]) + inc_size_x])
            plt.ylim([min([n.y for n in nodes]) - inc_size_y, max([n.y for n in nodes]) + inc_size_y])

            # plt.scatter(nodes_x, nodes_y, facecolors='cornflowerblue', marker=MarkerStyle('o', fillstyle='top'),
            #             s=size_profits)
            # plt.scatter(nodes_x, nodes_y, facecolors='salmon', marker=MarkerStyle('o', fillstyle='bottom'), s=size_weights)
            #
            # plt.scatter([nodes_x[0]], [nodes_y[0]], facecolors='darkslategray', marker='*', s=1000, zorder=10,
            #             path_effects=[pe.withStroke(linewidth=6, foreground='w')])

            nodes_x2 = []
            nodes_y2 = []
            for x in range(10):
                for y in range(5):
                    nodes_x2.append(6.5 * x + 5)
                    nodes_y2.append(12.5 * y + 10)
            # nodes_x2 = np.array([50.0] * 50)
            # nodes_y2 = np.array([50.0] * 50)
            size_profits2 = size_profits[1:]
            size_weights2 = size_weights[1:]

            plt.scatter(nodes_x2, nodes_y2, facecolors='cornflowerblue', marker=MarkerStyle('o', fillstyle='top'),
                        s=size_profits2)
            plt.scatter(nodes_x2, nodes_y2, facecolors='salmon', marker=MarkerStyle('o', fillstyle='bottom'), s=size_weights2)

            if with_items:
                items_x = [nodes_x2[i] for i in with_items]
                items_y = [nodes_y2[i] for i in with_items]
                plt.scatter(items_x, items_y, edgecolors='mediumseagreen', marker='o', facecolor='none', s=12000, linewidths=7, linestyle='--')
        else:
            plt.scatter(nodes_x, nodes_y, facecolors='cornflowerblue', marker='o', s=2000)

        if with_neighbors:
            self._plot_neighborhood(with_neighbors, nodes_x, nodes_y)

        if show_index:
            if isinstance(show_index, list):
                if isinstance(show_index[0], bool):
                    def item_label(item):
                        label = sum([1 for i in item.node.items if show_index[i.i]])
                        if label == 1:
                            label = item.node.label
                        return label

                    show_index = [(item.node, item_label(item)) for item in
                                  self.items if show_index[item.i]]
                else:
                    show_index = [Problem.find_node_by_label(nodes, label) for label in show_index]
            else:
                show_index = nodes

            if normalize_by:
                ratio = 100 / normalize_by[0]
            else:
                ratio = 1

            font_size = 22
            font_stroke_size = 4

            for node in show_index:
                if isinstance(node, tuple):
                    label = node[1]
                    node = node[0]
                else:
                    label = node.i + 1

                if label:
                    plt.annotate(label, (node.x, node.y), ha='center', va='center', size=font_size,
                                 path_effects=[pe.withStroke(linewidth=font_stroke_size, foreground='w')])

        plt.axis('off')

        return fig, ax

    def plot(self, nodes=None, show_index=True, normalize_by=None, with_neighbors=False, with_routes=None, with_route=None,
             savefile=False, show_items=True, figsize=None, run_after=None):
        fig, ax = self.plot_subgraph(nodes, show_index, normalize_by, with_neighbors, show_items, figsize=figsize)

        if with_route and not with_routes:
            with_routes = [with_route]

        if not with_routes:
            with_routes = []

        for with_route in with_routes:
            plt.plot([node.x for node in with_route], [node.y for node in with_route], color='darkslategray',
                     linestyle='-', linewidth=4)

        if run_after:
            run_after(self)

        fig.tight_layout()
        plt.show()

        if savefile:
            filename = f'images/{self.filename}'
            if nodes:
                filename += '_nodes'
            fig.savefig(filename, transparent=True, dpi=300)

        return fig, ax

    def plot_ks(self, nodes=None, show_index=True, normalize_by=None, with_neighbors=False, with_route=None,
             savefile=False, show_items=True, figsize=None, with_items=[]):
        fig, ax = self.plot_subgraph_ks(nodes, show_index, normalize_by, with_neighbors, show_items, figsize=figsize, with_items=with_items)

        if with_route:
            plt.plot([node.x for node in with_route], [node.y for node in with_route], color='darkslategray',
                     linestyle='--', linewidth=4)

        fig.tight_layout()
        plt.show()

        if savefile:
            filename = f'images/{self.filename}'
            if nodes:
                filename += '_nodes'
            fig.savefig(filename, transparent=True, dpi=300)

        return fig, ax

    def get_points(self):
        if self.points is None:
            self.points = np.array([(node.x, node.y) for node in self.nodes])

        return self.points

    def get_delaunay(self):
        if self.delaunay is not None:
            return self.delaunay

        delaunay = Delaunay(self.get_points())

        def find_neighbors(node):
            neighborhood_start = delaunay.vertex_neighbor_vertices[0][node]
            neighborhood_end = delaunay.vertex_neighbor_vertices[0][node + 1]

            neighborhood = delaunay.vertex_neighbor_vertices[1][neighborhood_start:neighborhood_end]
            neighborhood = [self.nodes[neighbor] for neighbor in neighborhood]

            node = self.nodes[node]
            neighborhood.sort(key=lambda n: node.distance(n))

            return [n.i for n in neighborhood if n.i]

        self.delaunay = [find_neighbors(node) for node in range(0, len(self.nodes))]

        return self.delaunay

    def plot_voronoi(self, show_index=None, normalize_by=None, with_neighbors=False, save_at=None, figsize=None, after_run=None):
        fig, ax = self.plot_subgraph(nodes=None, show_index=show_index, normalize_by=normalize_by, with_neighbors=with_neighbors, show_items=False, figsize=figsize)

        points = self.get_points()
        vor = Voronoi(points)
        ax.set_aspect('equal', adjustable='box')
        delaunay = Delaunay(points)
        plt.triplot(points[:, 0], points[:, 1], delaunay.simplices, color='darkslategray', linestyle='-', linewidth=2)
        voronoi_plot_2d(vor, ax, line_width=2, line_colors='darkgray', show_points=False, show_vertices=False)
        # plt.scatter(points[:, 0], points[:, 1], facecolors='cornflowerblue', marker='o', s=100)

        if after_run:
            after_run(self)

        plt.axis('off')
        fig.tight_layout()

        if save_at:
            plt.savefig(save_at, bbox_inches="tight", transparent=True, dpi=144)

        plt.show()

        return vor

    def plot_delaunay(self, show_index=None, normalize_by=None, with_neighbors=False, save_at = None):
        fig, ax = self.plot_subgraph(nodes=None, show_index=show_index, normalize_by=normalize_by, with_neighbors=with_neighbors, show_items=False)

        points = self.get_points()
        delaunay = Delaunay(points)

        ax.set_aspect('equal', adjustable='box')
        plt.triplot(points[:, 0], points[:, 1], delaunay.simplices, color='darkslategray', linestyle='-', linewidth=2)
        plt.scatter(points[:, 0], points[:, 1], facecolors='cornflowerblue', marker='o', s=300)

        # vextex = 0
        # for simplex in delaunay.simplices:
        #     for (p0, p1) in [(0, 1), (1, 2), (2, 0)]:
        #         p0 = points[simplex[p0]]
        #         p1 = points[simplex[p1]]
        #         plt.plot((p0[0], p1[0]), (p0[1], p1[1]), color='darkslategray', linestyle='-', linewidth=2)
        #         # print(p0, p0)

        plt.axis('off')
        fig.tight_layout()

        if save_at:
            plt.savefig(save_at, bbox_inches="tight", transparent=True, dpi=300)

        plt.show()

        return delaunay

    def get_matrix(self):
        if self.matrix is not None:
            return self.matrix

        matrix = []

        for row_node in self.nodes:
            row = []
            matrix.append(row)

            for col_node in self.nodes:
                row.append(row_node.distance(col_node))

        from scipy.sparse import lil_matrix
        self.matrix = lil_matrix(matrix)

        return self.matrix

    def get_weight_matrix(self):
        if self.matrix is not None:
            return self.weight_matrix

        weight = 0.0
        matrix = []

        for row_node in self.nodes:
            for item in row_node.items:
                if self.items[item.i]:
                    weight += item.weight

            speed = self.max_speed - weight * self.speed_coefficient

            row = []
            matrix.append(row)

            for col_node in self.nodes:
                if row_node.label == col_node.label:
                    dist = 999999
                else:
                    dist = row_node.distance(col_node) / speed, 6

                row.append(dist)

        self.weight_matrix = matrix
        return matrix
