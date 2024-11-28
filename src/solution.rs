use std::fmt;
use std::fmt::Formatter;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::iter::once;
use std::path::PathBuf;
use std::simd::{LaneCount, Simd, StdFloat, SupportedLaneCount};

use anyhow::{Context, Result};
use bitvec::prelude::*;

use crate::Env;
use crate::problem::{Item, Node, Problem};

#[derive(Clone)]
pub struct Solution<'a> {
    // Fitness of this solution
    pub fitness: f64,

    // Problem
    pub(crate) problem: &'a Problem,

    // Speed coefficient used to map a weight into speed
    speed_coefficient: f64,

    // Total profit
    pub(crate) profit: f64,

    // Total weight. Can't be larger than the problem's knapsack capacity
    pub(crate) weight: f64,

    // Starting node
    pub(crate) home_node: &'a Node,

    // Route for this solution
    pub(crate) route: Vec<&'a Node>,

    // Route for this solution
    pub(crate) positions: Vec<usize>,

    // Picking plan for this solution
    pub(crate) loot: BitVec,

    // Weights in each node of the route. Indexed by the node index.
    pub(crate) weights: Vec<f64>,

    // Cumulative weights and times in each node of the route. Indexed by the route index.
    pub(crate) cumulative: Vec<(f64, f64)>,
}

impl<'a> Solution<'a> {
    fn new(problem: &'a Problem, route: Vec<&'a Node>, loot: BitVec) -> Self {
        let mut res = Self {
            fitness: f64::NEG_INFINITY,
            problem,
            speed_coefficient: problem.speed_coefficient(),
            profit: 0.0,
            weight: 0.0,
            weights: vec![0.0; problem.nodes.len()],
            home_node: &problem.nodes[0],
            positions: vec![0; route.len() + 1],
            route,
            loot,
            cumulative: vec![(0.0, 0.0); problem.nodes.len()],
        };

        res.update_positions();

        res
    }

    pub fn from_route_and_loot(problem: &'a Problem, route: Vec<usize>, loot: Vec<usize>) -> Result<Self> {
        let expected_route_len = problem.nodes.len();

        ensure!(route.len() == expected_route_len, "Route must have the same length as in the problem. Expected {}, found {}.", expected_route_len, route.len());
        ensure!(route[0] == 1, "Route must start at node 1, found {}.", route[0]);

        for (pos, &node) in route.iter().enumerate() {
            ensure!(node >= 1 && node <= expected_route_len, "Nodes must be in range 1 to {expected_route_len}. Found {node} at position {}.", pos + 1);
        }

        for node in 1..problem.nodes.len() {
            ensure!(route.contains(&node), "Node {node} not found in route. Route must contain all nodes from 1 to {expected_route_len}.");
        }

        let expected_loot_range = problem.items.len();

        ensure!(loot.is_sorted(), "Items must be sorted.");

        let mut sol_loot = bitvec![0; problem.items.len()];

        let mut item_positions = vec![usize::MAX; *problem.items.iter().max().unwrap_or(&0) + 1];
        for (index, &item) in problem.items.iter().enumerate() {
            item_positions[item] = index;
        }

        for (pos, &item) in loot.iter().enumerate() {
            ensure!(item >= 1 && item <= expected_loot_range, "Items must be in range 1 to {expected_loot_range}. Found {item} at position {}.", pos + 1);

            let item_count = loot.iter().filter(|&&i| i == item).count();
            ensure!(item_count == 1, "Items must not repeat. Found `{item}`, {item_count} times starting at position {}.", pos + 1);

            let item_index = item_positions[item];
            ensure!(item_index != usize::MAX,  "Couldn't find the item `{item}` in problem.");

            sol_loot.set(item_positions[item], true);
        }

        let route = route.iter().skip(1).map(|&n| &problem.nodes[n - 1]).collect();

        let mut solution = Solution::new(problem, route, sol_loot);
        solution.update_weights_profit_and_fitness();
        solution.update_cumulative_data(0);

        ensure!(solution.weight <= solution.problem.capacity_of_knapsack,
            "Knapsack over capacity. Expected weight to at be most {}, got {}.", solution.problem.capacity_of_knapsack, solution.weight);

        Ok(solution)
    }

    pub fn from_lk_route(problem: &'a Problem, route: Vec<&'a Node>) -> Self {
        debug_assert_eq!(route[0].index, 0, "TSP route must start from node 0.");
        debug_assert_eq!(route[route.len() - 1].index, 0, "TSP route must end at node 0.");

        let route = route.iter()
            .copied()
            .skip(1)
            .take(route.len() - 2)
            .collect();

        Self::new(problem, route, bitvec![0; problem.items.len()])
    }

    pub fn from_file(problem: &'a Problem, input_file_path: &PathBuf, extra_info_check: bool) -> Result<(Self, Option<f64>)> {
        let input_file = File::open(input_file_path)
            .context(format!("Couldn't open solution file `{}`.", input_file_path.display()))?;
        let mut reader = BufReader::new(input_file);
        let mut line_string = String::with_capacity(200);

        let mut route = vec![];
        let mut loot = vec![];

        let mut line_counter = 0;

        while let Ok(bytes_read) = reader.read_line(&mut line_string) {
            if bytes_read == 0 {
                break;
            }

            line_counter += 1;
            if line_counter > 2 {
                continue;
            }

            let line = line_string.trim_end();

            let is_content_line = line.starts_with('[') && line.ends_with(']');
            ensure!(is_content_line, "Solution file's lines should start with `[` and end with `]`.");

            let line = &line[1..line.len() - 1];
            if !line.is_empty() {
                let line_split = line.split(',');

                let line = if line_counter == 1 { &mut route } else { &mut loot };

                for elem in line_split {
                    let elem = elem.trim();
                    let elem = elem.parse::<usize>()
                        .context(format!("Solution file's lines should be integers separated by commas. Found `{elem}` at line {line_counter}."))?;
                    line.push(elem);
                }
            }

            line_string.clear();
        }

        ensure!(line_counter == 2, "Solution file should contain exactly 2 lines, found {line_counter}.");

        let result = Solution::from_route_and_loot(problem, route, loot)?;
        let mut extra_fitness: Option<f64> = None;

        if extra_info_check {
            let log_file_path = &mut input_file_path.clone();
            let extension = log_file_path.extension()
                .and_then(|ext| ext.to_str())
                .map_or("log".to_owned(), |ext| format!("{ext}.log"));
            log_file_path.set_extension(extension);

            let log_file_path = &log_file_path;
            let log_file = File::open(log_file_path)
                .context(format!("Couldn't open log file `{}`.", log_file_path.display()))?;

            let mut reader = BufReader::new(log_file);

            line_string.clear();
            while let Ok(bytes_read) = reader.read_line(&mut line_string) {
                if bytes_read == 0 {
                    break;
                }

                if let Some(line_extra_fitness) = line_string.strip_prefix("fitness:") {
                    let line_extra_fitness = line_extra_fitness.trim();
                    let line_extra_fitness = line_extra_fitness.parse()
                        .context(format!("Couldn't parse `{line_extra_fitness}` as a number while reading fitness."))?;
                    extra_fitness = Some(line_extra_fitness);
                }

                line_string.clear();
            }
            let Some(extra_fitness) = extra_fitness else { bail!("Couldn't find the fitness value in log file.") };
            ensure!(extra_fitness == result.fitness, "Wrong fitness. Expected {}, got {extra_fitness}.", result.fitness);
        }

        Ok((result, extra_fitness))
    }

    fn check_route_is_valid(&self) {
        assert_eq!(self.route.len(), self.problem.nodes.len() - 1, "Invalid route. Route should always have one less node than the number of nodes in the problem.");

        let mut indexes = self.route.iter().map(|n| n.index).collect::<Vec<_>>();
        indexes.sort();

        for (mut expected, &actual) in indexes.iter().enumerate() {
            expected += 1;
            assert_eq!(actual, expected, "Invalid route. Missing node {} in route.", expected);
        }
    }

    pub(crate) fn debug_check_route_is_valid(&self) {
        if cfg!(debug_assertions) {
            self.check_route_is_valid();
        }
    }

    fn check_solution_is_valid(&self) {
        let (fitness, _, weight) = self.fitness_time_and_weight();

        assert!(weight <= self.problem.capacity_of_knapsack, "Invalid picking plan. Expected total weight to be at most {}, got {weight}", self.problem.capacity_of_knapsack);
        debug_assert_eq!(weight, self.weight, "Invalid weight. Expected weight be {}, got {weight}", self.weight);

        assert!((fitness - self.fitness).abs() < 0.0000001, "Invalid fitness. Expected {fitness}, got: {}", self.fitness);

        self.check_route_is_valid();
    }

    pub(crate) fn debug_check_solution_is_valid(&self) {
        if cfg!(debug_assertions) {
            self.check_solution_is_valid();
        }
    }

    pub(crate) fn update_positions(&mut self) {
        for (index, &node) in self.route.iter().enumerate() {
            self.positions[node.index] = index;
        }
    }

    pub(crate) fn update_positions_from_to(&mut self, from_index: usize, to_index: usize) {
        for (index, &node) in self.route[from_index..=to_index].iter().enumerate() {
            self.positions[node.index] = from_index + index;
        }
    }

    #[inline]
    pub(crate) fn weight_and_time_for_nodes(&self, previous_node: &Node, node: &Node, mut weight: f64, mut time: f64) -> (f64, f64) {
        weight += self.weights[previous_node.index];

        let dist = previous_node.distance(node);
        let speed = self.speed_coefficient.mul_add(-weight, self.problem.max_speed);

        time += dist / speed;

        (weight, time)
    }

    pub(crate) fn update_cumulative_data(&mut self, from_node: usize) -> f64 {
        let ((mut weight, mut time), mut previous_node) = if from_node == 0 {
            ((0.0, 0.0), self.home_node)
        } else {
            (self.cumulative[from_node - 1], self.route[from_node - 1])
        };

        for (index, &node) in self.route[from_node..].iter().enumerate() {
            (weight, time) = self.weight_and_time_for_nodes(previous_node, node, weight, time);
            previous_node = node;

            self.cumulative[index + from_node] = (weight, time);
        }

        (_, time) = self.weight_and_time_for_nodes(previous_node, self.home_node, weight, time);

        self.cumulative[self.route.len()] = (weight, time);

        self.problem.renting_ratio.mul_add(-time, self.profit)
    }

    pub(crate) fn fitness_with_weights(&self) -> f64 {
        let mut weight = 0.0;
        let mut time = 0.0;

        let mut previous_node = self.home_node;

        for &node in &self.route {
            (weight, time) = self.weight_and_time_for_nodes(previous_node, node, weight, time);
            previous_node = node;
        }
        (_, time) = self.weight_and_time_for_nodes(previous_node, self.home_node, weight, time);

        self.problem.renting_ratio.mul_add(-time, self.profit)
    }

    #[inline]
    pub(crate) fn weight_and_time_for_nodes_simd<const N: usize>(&self, previous_node: &Node, node: &Node, mut weight: Simd<f64, N>, mut time: Simd<f64, N>) -> (Simd<f64, N>, Simd<f64, N>)
        where LaneCount<N>: SupportedLaneCount {
        weight += Simd::splat(self.weights[previous_node.index]);

        let dist = previous_node.distance(node);
        let speed = Simd::splat(self.speed_coefficient).mul_add(-weight, Simd::splat(self.problem.max_speed));

        time += Simd::splat(dist) / speed;

        (weight, time)
    }

    #[inline]
    pub(crate) fn fitness_with_weights_changed<const N: usize>(&self, route_index: usize, profit_changes: [f64; 64], weight_changes: [f64; 64], fitness_candidate: &mut [f64; 64])
        where LaneCount<N>: SupportedLaneCount {
        let cumulative_weight_and_time = self.cumulative[route_index];

        let mut weight = Simd::<f64, N>::splat(cumulative_weight_and_time.0);
        let mut time = Simd::<f64, N>::splat(cumulative_weight_and_time.1);

        weight += Simd::<f64, N>::from_slice(&weight_changes);

        let mut previous_node = self.route[route_index];

        for &node in &self.route[route_index + 1..] {
            (weight, time) = self.weight_and_time_for_nodes_simd(previous_node, node, weight, time);
            previous_node = node;
        }

        (_, time) = self.weight_and_time_for_nodes_simd(previous_node, self.home_node, weight, time);

        let fitness = Simd::<f64, N>::from_slice(&profit_changes) +
            Simd::<f64, N>::splat(self.problem.renting_ratio).mul_add(-time, Simd::<f64, N>::splat(self.profit));
        fitness.copy_to_slice(fitness_candidate);
    }

    pub(crate) fn update_fitness_with_weights(&mut self) {
        self.fitness = self.fitness_with_weights();
    }

    pub(crate) fn time_with_candidate(&self, mut previous_node: &'a Node, mut weight: f64, mut time: f64, candidate: impl Iterator<Item=&'a Node>, next_node: usize) -> f64 {
        for node in candidate {
            (weight, time) = self.weight_and_time_for_nodes(previous_node, node, weight, time);
            previous_node = node;
        }

        let next_node = if next_node < self.route.len() {
            self.route[next_node]
        } else {
            self.home_node
        };

        (_, time) = self.weight_and_time_for_nodes(previous_node, next_node, weight, time);

        time
    }

    fn fitness_time_and_weight_from_loot(&self, route: impl Iterator<Item=&'a Node>, loot: &BitVec) -> (f64, f64, f64) {
        let mut profit = 0.0;
        let mut weight = 0.0;
        let mut time = 0.0;

        let mut previous_node = self.home_node;
        for node in route.chain(once(self.home_node)) {
            for item in &previous_node.items {
                if loot[item.index] {
                    profit += item.profit;
                    weight += item.weight;
                }
            }

            let dist = previous_node.distance(node);
            let speed = self.speed_coefficient.mul_add(-weight, self.problem.max_speed);

            time += dist / speed;

            previous_node = node;
        }

        (self.problem.renting_ratio.mul_add(-time, profit), time, weight)
    }

    pub(crate) fn fitness_time_and_weight(&self) -> (f64, f64, f64) {
        self.fitness_time_and_weight_from_loot(self.route.iter().copied(), &self.loot)
    }

    pub(crate) fn update_weights_profit_and_fitness(&mut self) {
        self.profit = 0.0;
        self.weight = 0.0;

        for node in &self.problem.nodes {
            let mut node_weight = 0.0;

            for item in &node.items {
                if self.loot[item.index] {
                    self.profit += item.profit;
                    self.weight += item.weight;
                    node_weight += item.weight;
                }
            }

            self.weights[node.index] = node_weight;
        }

        self.update_fitness_with_weights();
    }

    pub(crate) fn pick_from_power(&self, power: i32, scores: &[(&Item, f64, f64)]) -> BitVec {
        let mut scores = scores.iter().map(|score| {
            (score.0, score.1.powi(power) * score.2)
        }).collect::<Vec<_>>();
        scores.sort_by(|a, b| b.1.total_cmp(&a.1));

        let mut loot = bitvec![0; self.loot.len()];

        let mut weight = 0.0;
        for (item, _) in scores {
            if weight + item.weight <= self.problem.capacity_of_knapsack {
                weight += item.weight;
                loot.set(item.index, true);

                // If the smallest item can't fit, no other item can.
                if self.weight + self.problem.smallest_weight > self.problem.capacity_of_knapsack {
                    break;
                }
            }
        }

        loot
    }

    pub(crate) fn pick_from_test_power<const REVERSED: bool>(&mut self) {
        let mut scores = Vec::with_capacity(self.problem.items.len());

        let mut next_node = self.home_node;
        let mut dist = 0.0;

        // Distance is calculated from the back. So, when reversed, iterate normally.
        if REVERSED {
            for &node in self.route.iter() {
                dist += node.distance(next_node);

                for item in &node.items {
                    scores.push((item, item.profit / item.weight, 1.0 / dist));
                }

                next_node = node;
            }
        } else {
            for &node in self.route.iter().rev() {
                dist += node.distance(next_node);

                for item in &node.items {
                    scores.push((item, item.profit / item.weight, 1.0 / dist));
                }

                next_node = node;
            }
        };

        let best_loot = (1..=10)
            .filter_map(|power| {
                let loot = self.pick_from_power(power, &scores);

                let (fitness_candidate, _, _) = if REVERSED {
                    self.fitness_time_and_weight_from_loot(self.route.iter().rev().copied(), &loot)
                } else {
                    self.fitness_time_and_weight_from_loot(self.route.iter().copied(), &loot)
                };

                (fitness_candidate > self.fitness).then_some((fitness_candidate, loot))
            })
            .max_by(|a, b| a.0.total_cmp(&b.0));

        if let Some((_, best_loot)) = best_loot {
            if REVERSED {
                self.route.reverse();
                self.update_positions();
            }
            self.loot = best_loot;
            self.update_weights_profit_and_fitness();
        }
    }

    pub(crate) fn partial_clone_from(&mut self, other: &Self) {
        self.fitness = other.fitness;
        self.profit = other.profit;
        self.weight = other.weight;

        self.route.clone_from(&other.route);
        self.positions.clone_from(&other.positions);
        self.loot.clone_from(&other.loot);
        self.weights.clone_from(&other.weights);
    }

    pub(crate) fn save_to_directory(&self, env: &mut Env) -> Result<()> {
        if self.fitness <= env.best_fitness {
            return Ok(());
        };

        let elapsed = env.timer.elapsed().as_secs_f64();

        if !env.args.silent {
            println!("\t> {} {elapsed}", self.fitness);
        }
        env.best_fitness = self.fitness;
        env.time_at_best_fitness = env.timer.elapsed().as_secs_f64();

        if env.args.dont_save_solution {
            return Ok(());
        }

        let write_error_context = || format!("Couldn't write to file {}.", env.output_file.display());

        let mut output = File::create(env.output_file.clone()).with_context(write_error_context)?;
        write!(output, "{self}").with_context(write_error_context)
    }
}

impl<'a> fmt::Display for Solution<'a> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "[1")?;
        for &node in &self.route {
            write!(f, ", {}", node.index + 1)?;
        }
        writeln!(f, "]")?;

        let mut is_first = true;
        let mut loot = self.problem.items.iter().enumerate()
            .filter_map(|(index, &label)| self.loot[index].then_some(label))
            .collect::<Vec<_>>();
        loot.sort_unstable();

        write!(f, "[")?;
        for label in loot {
            if is_first {
                write!(f, "{label}")?;
                is_first = false;
            } else {
                write!(f, ", {label}")?;
            }
        }
        writeln!(f, "]")
    }
}

// Fitness tests compare the results with https://cs.adelaide.edu.au/~ec/research/ttp/ObjComputation_TTP.cs
#[cfg(test)]
mod fitness {
    use std::path::PathBuf;

    use super::*;

    #[allow(dead_code)] // It's being used, but clippy complains anyway
    fn get_fitness(instance: &str, loot: &[(usize, usize)]) -> Result<f64> {
        let problem = Problem::new(&PathBuf::from(instance), &None).context("Error loading problem.")?;

        let route = problem.nodes.iter()
            .skip(1)
            .collect::<Vec<_>>();

        let mut solution_ref = Solution::new(&problem, route.clone(), bitvec![0; problem.items.len()]);
        let mut solution = Solution::new(&problem, route, bitvec![0; problem.items.len()]);

        for &(hop, pick) in loot {
            let node = solution_ref.route[hop - 1].index;
            let item = &solution_ref.problem.nodes[node].items[pick];

            solution_ref.loot.set(item.index, true);

            solution.loot.set(item.index, true);
            solution.profit += item.profit;
            solution.weight += item.weight;
            solution.weights[node] += item.weight;
        }

        solution_ref.update_weights_profit_and_fitness();
        let (fitness_ref, _, _) = solution_ref.fitness_time_and_weight();
        let fitness = solution.fitness_with_weights();
        ensure!(fitness == fitness_ref);
        Ok(fitness_ref)
    }

    #[test]
    fn a280_n1_bsc_fitness() {
        let expected = get_fitness("instances/a280_n279_bounded-strongly-corr_01.ttp", &[(1, 0), (5, 0)]).unwrap();
        let actual = 15396.1797553677;
        assert!((expected + actual).abs() < 0.000000001);
    }

    #[test]
    fn eil76_n3_bsc_fitness() {
        let expected = get_fitness("instances/eil76_n225_bounded-strongly-corr_01.ttp", &[(1, 0), (5, 0)]).unwrap();
        let actual = 56666.5513138363;
        assert!((expected + actual).abs() < 0.000000001);
    }

    #[test]
    fn eil76_n3_usw_09() {
        let expected = get_fitness("instances/eil76_n225_uncorr-similar-weights_09.ttp", &[(1, 0), (14, 1), (14, 2), (33, 2)]).unwrap();
        let actual = 169173.46511173;
        assert!((expected + actual).abs() < 0.000000001);
    }
}

#[cfg(test)]
mod check_is_valid {
    use clap::Parser;
    use crate::Args;
    use super::*;

    fn load_solution_and_problem(problem_file: &str, solution_file: &str, extra_info_check: bool) -> Result<f64> {
        let args = Args::parse_from(["test", problem_file]);
        let env = Env::from_args(args)?;
        let problem = Problem::new(&env.args.problem_file, &None)?;

        let solution_file = format!("tests/check/{solution_file}");
        let (solution, _) = Solution::from_file(&problem, &solution_file.into(), extra_info_check)?;

        Ok(solution.fitness)
    }

    fn load_solution(solution_file: &str) -> Result<f64> {
        load_solution_and_problem("tests/check/000_problem.ttp", solution_file, false)
    }

    #[test]
    #[should_panic(expected = "Couldn't open solution file")]
    fn file_not_found() {
        load_solution("000_file_not_found").unwrap();
    }

    #[test]
    #[should_panic(expected = "Solution file should contain exactly 2 lines, found 0.")]
    fn empty() {
        load_solution("001_empty").unwrap();
    }

    #[test]
    #[should_panic(expected = "Solution file should contain exactly 2 lines, found 1.")]
    fn single_line() {
        load_solution("002_single_line").unwrap();
    }

    #[test]
    #[should_panic(expected = "Solution file should contain exactly 2 lines, found 4.")]
    fn four_lines() {
        load_solution("003_four_lines").unwrap();
    }

    #[test]
    #[should_panic(expected = "Solution file's lines should start with `[` and end with `]`.")]
    fn no_starting_brackets() {
        load_solution("004_no_starting_brackets").unwrap();
    }

    #[test]
    #[should_panic(expected = "Solution file's lines should start with `[` and end with `]`.")]
    fn no_ending_brackets() {
        load_solution("005_no_ending_brackets").unwrap();
    }

    #[test]
    #[should_panic(expected = "Solution file's lines should start with `[` and end with `]`.")]
    fn both_brackets_missing() {
        load_solution("006_both_brackets_missing").unwrap();
    }

    #[test]
    #[should_panic(expected = "Solution file's lines should be integers separated by commas. Found `asdf` at line 1.")]
    fn route_with_no_numbers() {
        load_solution("007_route_with_no_numbers").unwrap();
    }

    #[test]
    #[should_panic(expected = "Solution file's lines should be integers separated by commas. Found `18; 19` at line 2.")]
    fn loot_with_no_numbers() {
        load_solution("008_loot_with_no_numbers").unwrap();
    }

    #[test]
    #[should_panic(expected = "Route must have the same length as in the problem. Expected 280, found 0.")]
    fn route_is_empty() {
        load_solution("020_route_is_empty").unwrap();
    }

    #[test]
    #[should_panic(expected = "Route must have the same length as in the problem. Expected 280, found 279.")]
    fn route_is_shorter() {
        load_solution("021_route_is_shorter").unwrap();
    }

    #[test]
    #[should_panic(expected = "Route must have the same length as in the problem. Expected 280, found 281.")]
    fn route_is_longer() {
        load_solution("022_route_is_longer").unwrap();
    }

    #[test]
    #[should_panic(expected = "Route must start at node 1, found 2.")]
    fn doesnt_start_at_1() {
        load_solution("023_doesnt_start_at_1").unwrap();
    }

    #[test]
    #[should_panic(expected = "Nodes must be in range 1 to 280. Found 0 at position 2.")]
    fn route_contains_zero() {
        load_solution("024_route_contains_zero").unwrap();
    }

    #[test]
    #[should_panic(expected = "Nodes must be in range 1 to 280. Found 281 at position 2.")]
    fn route_outside_range() {
        load_solution("025_route_outside_range").unwrap();
    }

    #[test]
    #[should_panic(expected = "Node 241 not found in route. Route must contain all nodes from 1 to 280.")]
    fn route_missing_node() {
        load_solution("026_route_missing_node").unwrap();
    }

    #[test]
    #[should_panic(expected = "Items must be sorted.")]
    fn items_not_sorted() {
        load_solution("040_items_not_sorted").unwrap();
    }

    #[test]
    #[should_panic(expected = "Items must be in range 1 to 279. Found 280 at position 68.")]
    fn items_outside_range() {
        load_solution("041_items_outside_range").unwrap();
    }

    #[test]
    #[should_panic(expected = "Items must not repeat. Found `18`, 3 times starting at position 5.")]
    fn item_repeated() {
        load_solution("042_item_repeated").unwrap();
    }

    #[test]
    #[should_panic(expected = "Knapsack over capacity. Expected weight to at be most 25936, got 32160.")]
    fn over_capacity() {
        load_solution("050_over_capacity").unwrap();
    }

    #[test]
    #[should_panic(expected = "Couldn't open log file")]
    fn extra_no_file() {
        load_solution_and_problem("tests/check/000_problem.ttp", "100_correct", true).unwrap();
    }

    #[test]
    #[should_panic(expected = "Couldn't find the fitness value in log file.")]
    fn extra_no_fitness() {
        load_solution_and_problem("tests/check/000_problem.ttp", "200_extra_no_fitness", true).unwrap();
    }

    #[test]
    #[should_panic(expected = "Couldn't parse `pretty bad :/` as a number while reading fitness.")]
    fn extra_fitness_not_number() {
        load_solution_and_problem("tests/check/000_problem.ttp", "201_extra_fitness_not_number", true).unwrap();
    }

    #[test]
    #[should_panic(expected = "Wrong fitness. Expected 19542.326368121983, got 195420.32636812198.")]
    fn extra_wrong_fitness() {
        load_solution_and_problem("tests/check/000_problem.ttp", "202_extra_wrong_fitness", true).unwrap();
    }

    #[test]
    fn correct() {
        let actual = load_solution("100_correct").unwrap();
        assert_eq!(actual, 19585.227149501345);
    }

    #[test]
    fn correct_no_items() {
        let actual = load_solution("101_correct_no_items").unwrap();
        assert_eq!(actual, -15135.78);
    }

    #[test]
    fn correct_multiple_items_per_city() {
        let actual = load_solution_and_problem("tests/check/000_problem_vm.ttp", "103_correct_multiple_items_per_city", false).unwrap();
        assert_eq!(actual, 2414366.98018748);
    }

    #[test]
    fn extra_correct() {
        let actual = load_solution_and_problem("tests/check/000_problem.ttp", "300_extra_correct", true).unwrap();
        assert_eq!(actual, 19542.326368121983);
    }

    #[test]
    fn extra_correct_with_extension() {
        let actual = load_solution_and_problem("tests/check/000_problem.ttp", "301_extra_correct.flns", true).unwrap();
        assert_eq!(actual, 19542.326368121983);
    }
}
