from __future__ import annotations

import json
import math
from dataclasses import dataclass

from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection

from lib.ttp.problem import Item


@dataclass()
class Score:
    item: Item
    value: float

    def __lt__(self, other):
        return self.value < other.value

    def __repr__(self):
        cls = self.__class__.__name__
        node = self.item.node.i + 1
        return f'{cls}: node={node}, value={self.value}, profit={self.item.profit}, weight={self.item.weight})'


@dataclass(init=False, eq=False)
class Solution:
    def __init__(self, problem, route):
        self.problem = problem
        self.route = route
        self.items = [False for _ in problem.items]

    def clone(self):
        cloned = Solution(self.problem, self.route[:])
        cloned.items = self.items[:]
        return cloned

    def clone_from(self, other):
        self.problem = other.problem
        self.route = list(other.route)
        self.items = list(other.items)

    @staticmethod
    def from_labels(problem, route, items):
        assert route[0] == 1, \
            f'First node should be fixed. Found `{route[0]}`, expected `1`'

        assert len(route) == len(problem.nodes), \
            f'Route length should be the same of the problem. Found `{len(route)}`, expected `{len(problem.nodes)}`'

        nodes_found = []
        for node in route:
            assert 0 < node <= len(problem.nodes), \
                f'Nodes should be between 1 and nodes\'s length. Found `{node}`'

            assert node not in nodes_found, \
                f'Nodes should not repeat in route. Found `{node}` more than once.'

            nodes_found.append(node)

        items_found = []
        for item in items:
            assert 0 < item <= len(problem.items), \
                f'Picking item should be between 1 and item\'s length. Found `{item}`'

            assert item not in items_found, \
                f'Items should not repeat in picking plan. Found `{item}` more than once.'

            items_found.append(item)

        route = [problem.nodes[node - 1] for node in route + [1]]
        solution = Solution(problem, route)
        for item in items:
            solution.items[item - 1] = True
        return solution

    @staticmethod
    def from_filename(problem, filename):
        with open(filename) as file:
            # file.readline()
            # file.readline()
            # file.readline()
            # file.readline()
            # file.readline()
            # file.readline()
            route = json.loads(file.readline())
            items = json.loads(file.readline())

        return Solution.from_labels(problem, route, items)

    @staticmethod
    def from_directory(problem, directory='results'):
        return Solution.from_filename(problem, f'{directory}/{problem.filename}.flns')

    def _evaluate_score(self, key):
        score = [key(item) for item in self.problem.items]
        score.sort(reverse=True)
        return score

    def node_index_by_label(self, label):
        for index, node in enumerate(self.route):
            if node.label == label:
                return index

    def find_node_by_label(self, label):
        return self.problem.find_node_by_label(self.route, label)

    def score_euclidean_dist(self):
        depot = self.route[0]

        def _score(item):
            dist = item.node.distance(depot)
            return Score(item, item.profit / item.weight / dist)

        return self._evaluate_score(_score)

    def _pick_from_scores(self, scores: [Score]):
        weight = 0.0

        for score in scores:
            item = score.item

            if weight + item.weight > self.problem.capacity_of_knapsack:
                self.items[item.i] = False
            else:
                self.items[item.i] = True
                weight += item.weight

        return self

    def _score_dist_in_route(self, power):
        score = []

        for node_index in range(0, len(self.route)):
            dist = 0.0
            for (prev_node, node) in zip(self.route[node_index:], self.route[node_index + 1:]):
                dist += prev_node.distance(node)

            node = self.route[node_index]
            for item in node.items:
                score.append(Score(item, math.pow(item.profit / item.weight, power) / dist))

        score.sort(reverse=True)

        return score

    def pick_from_dist_in_route(self):
        best_fitness = 0
        best_power = 0

        for power in range(10, 150):
            power /= 10

            score = self._score_dist_in_route(power)
            self._pick_from_scores(score)
            fitness = self.fitness()

            if fitness > best_fitness:
                best_fitness = fitness
                best_power = power

        score = self._score_dist_in_route(best_power)
        self._pick_from_scores(score)

    def fitness(self):
        profit = 0.0
        weight = 0.0
        time = 0.0

        for (prev_node, node) in zip(self.route, self.route[1:]):
            for item in prev_node.items:
                if self.items[item.i]:
                    profit += item.profit
                    weight += item.weight

            dist = prev_node.distance(node)
            speed = self.problem.max_speed - weight * self.problem.speed_coefficient

            time += dist / speed

        if weight > self.problem.capacity_of_knapsack:
            return profit - weight - 1000000

        return profit - self.problem.renting_ratio * time

    def distance_nodes_labels(self, label_from, label_to):
        if label_from == label_to:
            return 0.0

        node_from = self.find_node_by_label(label_from)
        node_to = self.find_node_by_label(label_to)

        route_from_index = self.route.index(node_from)
        route_to_index = self.route.index(node_to)
        sign = 1

        if route_from_index > route_to_index:
            route_from_index, route_to_index = route_to_index, route_from_index
            sign = -1

        dist = 0.0

        for node_index in range(route_from_index, route_to_index):
            prev_node = self.route[node_index]
            node = self.route[node_index + 1]
            dist += prev_node.distance(node)

        return sign * dist

    def weight(self):
        weight = 0.0

        for node in self.route:
            for item in node.items:
                if self.items[item.i]:
                    weight += item.weight

        return weight

    def weight_picked_at_node(self, node):
        weight = 0.0

        for item in node.items:
            if self.items[item.i]:
                weight += item.weight

        return weight

    def distance(self):
        dist = 0.0

        for (prev_node, node) in zip(self.route, self.route[1:]):
            dist += prev_node.distance(node)

        return dist

    def reverse_route(self, start, stop):
        size = stop + start
        for i in range(start, (size + 1) // 2):
            j = size - i
            self.route[i], self.route[j] = self.route[j], self.route[i]

    def local_search(self):
        best_fitness = self.fitness()
        has_improved = True

        while has_improved:
            has_improved = False

            # One flip
            for i in range(0, len(self.items)):
                self.items[i] = not self.items[i]

                fitness = self.fitness()
                if fitness > best_fitness:
                    has_improved = True
                    best_fitness = fitness
                else:
                    self.items[i] = not self.items[i]

            # Two flip
            # for i in range(0, len(self.items)):
            #     for j in range(0, len(self.items)):
            #         self.items[i] = not self.items[i]
            #         self.items[j] = not self.items[j]
            #
            #         fitness = self.fitness()
            #         if fitness > best_fitness:
            #             has_improved = True
            #             best_fitness = fitness
            #         else:
            #             self.items[i] = not self.items[i]
            #             self.items[j] = not self.items[j]

            # Move and flip?
            # for i in range(0, len(self.items)):
            #     for j in range(0, len(self.items)):
            #         for k in range(0, len(self.items)):
            #             self.items[i] = not self.items[i]
            #             self.items[j] = not self.items[j]
            #             self.items[k] = not self.items[k]
            #
            #             fitness = self.fitness()
            #             if fitness > best_fitness:
            #                 has_improved = True
            #                 best_fitness = fitness
            #             else:
            #                 self.items[i] = not self.items[i]
            #                 self.items[j] = not self.items[j]
            #                 self.items[k] = not self.items[k]

            # Two opt
            for i in range(1, len(self.route) - 2):
                for j in range(i + 1, len(self.route) - 1):
                    self.reverse_route(i, j)

                    fitness = self.fitness()
                    if fitness > best_fitness:
                        has_improved = True
                        best_fitness = fitness
                    else:
                        self.reverse_route(i, j)

            # One move
            for i in range(1, len(self.route) - 1):
                node = self.route.pop(i)
                for j in range(1, len(self.route)):
                    self.route.insert(j, node)
                    fitness = self.fitness()

                    if fitness > best_fitness:
                        has_improved = True
                        best_fitness = fitness
                        break
                    else:
                        self.route.pop(j)
                else:
                    self.route.insert(i, node)

            print(best_fitness)

    def change_route(self, *args):
        put = list(args)

        while len(put) > 1:
            if put[0] == put[1]:
                put.pop(0)
            else:
                break

        if len(put) <= 1:
            return self

        after = put.pop(0)

        after = self.problem.find_node_by_label(self.route, after)

        for node_label in put:
            node = self.problem.find_node_by_label(self.route, node_label)
            node_index = self.route.index(node)
            node = self.route.pop(node_index)

            after_index = self.route.index(after)
            self.route.insert(after_index + 1, node)

            after = node

        return self

    def _plot_route(self, ax, min_width=1, max_width=12):
        width_range = max_width - min_width

        weight = self.problem.capacity_of_knapsack

        line_segments = []
        line_widths = []

        for i in range(0, len(self.route) - 1):
            node = self.route[i]
            weight -= sum([item.weight for item in node.items if self.items[item.i]])

            width = min_width + width_range * weight / self.problem.capacity_of_knapsack

            line_segments.append(([node.x, node.y], [self.route[i + 1].x, self.route[i + 1].y]))
            line_widths.append(width)

        ax.add_collection(
            LineCollection(line_segments, linewidths=line_widths, color='darkslategray', capstyle='round'))

    def plot(self, normalize_by=None, with_neighbors=False, savefile=False, show_index='selected', min_line_width=1, max_line_width=12, show_items=True, figsize=(22, 10)):
        if show_index == 'selected':
            index_items = self.items
        elif show_index == 'all':
            index_items = [True for _ in self.items]
        else:
            index_items = []

        fig, ax = self.problem.plot_subgraph(nodes=self.route, show_index=index_items, normalize_by=normalize_by,
                                             with_neighbors=with_neighbors, show_items=show_items, figsize=figsize)

        self._plot_route(ax, min_width=min_line_width, max_width=max_line_width)

        plt.axis('off')
        fig.tight_layout()

        if savefile:
            fitness = str(self.fitness())
            fitness = fitness.replace('.', '-')

            if not isinstance(savefile, str):
                savefile = f'{self.problem.filename}_{fitness}'
            fig.savefig(f'images/{savefile}.png', transparent=True, dpi=300)

        return self

    def plot_route(self, normalize_by=None, with_neighbors=False, savefile=False, show_index='selected', max_line_width=12):
        # if show_index == 'selected':
        #     index_items = self.items
        # elif show_index == 'all':
        #     index_items = [True for _ in self.items]
        # else:
        #     index_items = []
        #
        index_items = []
        fig, ax = self.problem.plot_subgraph(nodes=self.route, show_index=index_items, normalize_by=normalize_by,
                                             with_neighbors=with_neighbors, show_items=False)

        self._plot_route(ax, max_width=max_line_width)
        #
        # plt.axis('off')
        # fig.tight_layout()
        #
        # if savefile:
        #     fitness = str(self.fitness())
        #     fitness = fitness.replace('.', '-')
        #
        #     if not isinstance(savefile, str):
        #         savefile = f'{self.problem.filename}_{fitness}'
        #     fig.savefig(f'images/{savefile}.png')

        return self

    def save_to_directory(self, directory='results'):
        nodes = [node.i + 1 for node in self.route[:-1]]
        items = [item.i + 1 for item in self.problem.items if self.items[item.i]]

        with open(f'{directory}/{self.problem.filename}.sol', 'w') as file:
            file.writelines([str(nodes), '\n', str(items)])

    def get_matrix(self):
        matrix = []

        for row_node in self.route[:-1]:
            row = []
            matrix.append(row)

            for col_node in self.route[:-1]:
                row.append(row_node.distance(col_node))

        return matrix

    def get_weight_matrix(self):
        weight = 0.0
        matrix = []

        for row_node in self.route[:-1]:
            weight += self.weight_picked_at_node(row_node)

            row = []
            matrix.append(row)

            for col_node in self.route[:-1]:
                if row_node.label == col_node.label:
                    dist = 999999
                else:
                    dist = row_node.distance(col_node)
                row.append(dist)

        return matrix
