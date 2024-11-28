#![feature(is_sorted)]
#![feature(portable_simd)]
#![feature(sort_floats)]
#![feature(iter_collect_into)]

#![warn(clippy::all)]

#![deny(clippy::correctness)]

#![warn(clippy::disallowed_types)]

#![warn(clippy::assertions_on_result_states)]
#![warn(clippy::dbg_macro)]
#![warn(clippy::decimal_literal_representation)]
#![warn(clippy::empty_structs_with_brackets)]
#![warn(clippy::exit)]
#![warn(clippy::expect_used)]
#![warn(clippy::float_cmp_const)]
#![warn(clippy::if_then_some_else_none)]
#![warn(clippy::let_underscore_must_use)]
#![warn(clippy::mutex_atomic)]
#![warn(clippy::panic)]
#![warn(clippy::panic_in_result_fn)]
#![warn(clippy::rc_mutex)]
#![warn(clippy::rest_pat_in_fully_bound_structs)]
#![warn(clippy::str_to_string)]
#![warn(clippy::string_to_string)]
#![warn(clippy::todo)]
#![warn(clippy::undocumented_unsafe_blocks)]
#![warn(clippy::unimplemented)]
#![warn(clippy::unwrap_in_result)]
#![warn(clippy::unwrap_used)]
#![warn(clippy::use_debug)]
#![warn(clippy::verbose_file_reads)]
#![warn(clippy::wildcard_enum_match_arm)]

#[macro_use]
extern crate anyhow;

use std::cmp::Ordering;
use std::hint::unreachable_unchecked;
use std::mem::swap;
use std::simd::{LaneCount, SupportedLaneCount};

use anyhow::{Context, Result};
use bitvec::prelude::*;
use rand::prelude::{IteratorRandom, SliceRandom};
use rand::Rng;
use rand_xoshiro::Xoshiro256PlusPlus;

pub use crate::env::{Args, Env};
use crate::env::RecordType::{InitialSolution, ImproveHeuristic, StartHeuristic};
use crate::problem::Node;
pub use crate::problem::Problem;
pub use crate::solution::Solution;

pub mod problem;
pub(crate) mod solution;
pub mod linkernighan;
pub(crate) mod env;

#[repr(usize)]
#[derive(Copy, Clone, Debug)]
enum SubNeighborhoodType {
    TwoOpt = 0,
    OneMove = 1,
}

#[derive(Clone)]
pub struct SubNeighborhood {
    route: [BitVec; 2],
    loot: BitVec,
}

impl SubNeighborhood {
    pub fn new(problem: &Problem) -> Self {
        let mut res = Self {
            route: [bitvec![1; problem.nodes.len()], bitvec![1; problem.nodes.len()]],
            loot: bitvec![1; problem.items.len()],
        };

        res.reset();

        res
    }

    fn reset_to(&mut self, value: bool) {
        for sub_neighborhood in self.route.iter_mut() {
            sub_neighborhood.fill(value);
            sub_neighborhood.set(0, false);
        }
    }

    fn reset(&mut self) {
        self.reset_to(true);
    }

    #[inline]
    fn is_route_active(&self, sn_type: SubNeighborhoodType, node: usize) -> bool {
        self.route[sn_type as usize][node]
    }

    #[inline]
    fn activate_node(&mut self, node: usize) {
        for sub_neighborhood in self.route.iter_mut() {
            sub_neighborhood.set(node, true);
        }
    }

    #[inline]
    fn deactivate_node(&mut self, sn_type: SubNeighborhoodType, node: usize) {
        self.route[sn_type as usize].set(node, false);
    }

    fn debug_assert_no_route_active(&self, sn_type: SubNeighborhoodType, env: &Env) {
        if env.should_continue() {
            let route = &self.route[sn_type as usize];
            debug_assert!(route.not_any(), "Sub-neighborhood {sn_type:?} shouldn't have any active nodes, is {}.", route);
        }
    }

    fn debug_assert_no_loot_active(&self, env: &Env) {
        if env.should_continue() {
            debug_assert!(self.loot.not_any(), "No loot should be active, is {}.", &self.loot);
        }
    }
}

trait RouteSearch: Sync {
    fn time(solution: &Solution, i: usize, j: usize, previous_node: &Node, weight: f64, time: f64) -> f64;

    fn apply(solution: &mut Solution, i: usize, j: usize);

    fn sub_neighborhood_type() -> SubNeighborhoodType;
}

struct TwoOpt;

impl RouteSearch for TwoOpt {
    fn time(solution: &Solution, i: usize, j: usize, previous_node: &Node, weight: f64, time: f64) -> f64 {
        let route_candidate = solution.route[i..=j].iter().rev().copied();
        solution.time_with_candidate(previous_node, weight, time, route_candidate, j + 1)
    }

    fn apply(solution: &mut Solution, i: usize, j: usize) {
        solution.route[i..=j].reverse();
    }

    fn sub_neighborhood_type() -> SubNeighborhoodType {
        SubNeighborhoodType::TwoOpt
    }
}

struct OneMove;

impl RouteSearch for OneMove {
    fn time(solution: &Solution, i: usize, j: usize, previous_node: &Node, weight: f64, time: f64) -> f64 {
        let route_candidate = solution.route[j..=j].iter().copied()
            .chain(solution.route[i..j].iter().copied());

        solution.time_with_candidate(previous_node, weight, time, route_candidate, j + 1)
    }

    fn apply(solution: &mut Solution, i: usize, j: usize) {
        solution.route[i..=j].rotate_right(1);
    }

    fn sub_neighborhood_type() -> SubNeighborhoodType {
        SubNeighborhoodType::OneMove
    }
}

fn fast_local_search_route<SEARCH: RouteSearch>(solution: &mut Solution, sub_neighborhood: &mut SubNeighborhood, env: &Env) -> bool {
    if !env.should_continue() {
        return false;
    }

    let sub_neighborhood_type = SEARCH::sub_neighborhood_type();

    let mut has_improved = false;

    let mut previous_node_i = solution.home_node;
    let mut previous_weight_i = 0.0;
    let mut previous_time_i = 0.0;

    for i in 0..solution.route.len() - 1 {
        let node_i = solution.route[i];

        if sub_neighborhood.is_route_active(sub_neighborhood_type, node_i.index) {
            let candidates_move = previous_node_i.adjacency.iter()
                .filter_map(|&node_j_index| {
                    let j = solution.positions[node_j_index];
                    if j <= i {
                        return None;
                    }

                    let time_candidate = SEARCH::time(solution, i, j, previous_node_i, previous_weight_i, previous_time_i);
                    let time_diff = solution.cumulative[j + 1].1 - time_candidate;
                    let rent_diff = time_diff * solution.problem.renting_ratio;
                    (rent_diff > 1e-10).then_some((j, rent_diff))
                })
                .max_by(|a, b|
                    match a.1.total_cmp(&b.1) {
                        Ordering::Equal => a.0.cmp(&b.0).reverse(),
                        Ordering::Less => Ordering::Less,
                        Ordering::Greater => Ordering::Greater,
                    }
                );

            if let Some((improved_index, rent_diff)) = candidates_move {
                let node_improved = solution.route[improved_index];
                let node_after_improved = if improved_index < solution.route.len() - 1 {
                    solution.route[improved_index + 1]
                } else {
                    node_improved
                };

                for node_sub_neighbor in [node_improved, node_after_improved] {
                    let position_activating = solution.positions[node_sub_neighbor.index];
                    debug_assert!(position_activating > i, "Trying to activate passed node {position_activating}, current position is {i}.");
                    sub_neighborhood.activate_node(node_sub_neighbor.index);
                }

                SEARCH::apply(solution, i, improved_index);
                solution.update_positions_from_to(i, improved_index);

                let new_fitness = solution.update_cumulative_data(i);
                debug_assert!(new_fitness > solution.fitness, "Expected new fitness to be greater than {}. Got {new_fitness}. Rent diff is {}.", solution.fitness, rent_diff);
                solution.fitness = new_fitness;

                has_improved = true;

                solution.debug_check_solution_is_valid();
            }

            let node_i = solution.route[i];
            sub_neighborhood.deactivate_node(sub_neighborhood_type, node_i.index);
        }

        previous_node_i = solution.route[i];

        (previous_weight_i, previous_time_i) = solution.cumulative[i];
    }

    // Last node will not be iterated
    sub_neighborhood.deactivate_node(sub_neighborhood_type, solution.route[solution.route.len() - 1].index);

    sub_neighborhood.debug_assert_no_route_active(sub_neighborhood_type, env);
    solution.debug_check_solution_is_valid();
    has_improved
}

fn one_flip<const MAX_ITEMS_PER_CITY: usize, const REMOVING: bool>(solution: &mut Solution, sub_neighborhood: &mut SubNeighborhood, env: &Env) -> bool
    where LaneCount<MAX_ITEMS_PER_CITY>: SupportedLaneCount {
    if !REMOVING && solution.weight + solution.problem.smallest_weight > solution.problem.capacity_of_knapsack {
        if cfg!(debug_assertions) {
            for &item in &solution.problem.items {
                let item = item - 1;
                if solution.loot[item] == REMOVING {
                    sub_neighborhood.loot.set(item, false);
                }
            }
        }

        return false;
    }

    if !env.should_continue() {
        return false;
    }

    let mut profit_changes = [0.0; 64];
    let mut weight_changes = [0.0; 64];
    let mut fitness_candidate = [0.0; 64];

    let mut candidates = vec![];

    let mut has_improved = false;

    'outer: for i in 0..solution.route.len() {
        let i = if REMOVING {
            i
        } else {
            solution.route.len() - 1 - i
        };

        let node_i = solution.route[i];

        'candidates: loop {
            candidates.clear();

            for item_index in 0..node_i.items.len() {
                let item = &node_i.items[item_index];
                if solution.loot[item.index] != REMOVING {
                    continue;
                }

                if !sub_neighborhood.loot[item.index] {
                    continue;
                }

                let (profit_change, weight_change) = if REMOVING {
                    (-item.profit, -item.weight)
                } else {
                    if solution.weight + item.weight > solution.problem.capacity_of_knapsack {
                        sub_neighborhood.loot.set(item.index, false);
                        continue;
                    }

                    (item.profit, item.weight)
                };

                profit_changes[candidates.len()] = profit_change;
                weight_changes[candidates.len()] = weight_change;

                candidates.push(item.index);
            }

            if candidates.is_empty() {
                break;
            }

            let simd_implementation = candidates.len().next_power_of_two();

            match simd_implementation {
                0 => {
                    // SAFETY: `next_power_of_two` is always positive.
                    unsafe { unreachable_unchecked() }
                }
                1 => solution.fitness_with_weights_changed::<1>(i, profit_changes, weight_changes, &mut fitness_candidate),
                2 => solution.fitness_with_weights_changed::<2>(i, profit_changes, weight_changes, &mut fitness_candidate),
                4 => solution.fitness_with_weights_changed::<4>(i, profit_changes, weight_changes, &mut fitness_candidate),
                8 => solution.fitness_with_weights_changed::<8>(i, profit_changes, weight_changes, &mut fitness_candidate),
                16 => solution.fitness_with_weights_changed::<16>(i, profit_changes, weight_changes, &mut fitness_candidate),
                32 => solution.fitness_with_weights_changed::<32>(i, profit_changes, weight_changes, &mut fitness_candidate),
                _ => solution.fitness_with_weights_changed::<64>(i, profit_changes, weight_changes, &mut fitness_candidate),
            };

            let mut next_candidate = 0;
            while fitness_candidate[next_candidate] <= solution.fitness {
                sub_neighborhood.loot.set(candidates[next_candidate], false);
                next_candidate += 1;

                if next_candidate >= candidates.len() {
                    break 'candidates;
                }
            }
            sub_neighborhood.loot.set(candidates[next_candidate], false);

            let item_index = candidates[next_candidate];
            let profit_change = profit_changes[next_candidate];
            let weight_change = weight_changes[next_candidate];

            let loot_value = solution.loot[item_index];
            solution.loot.set(item_index, !loot_value);

            solution.profit += profit_change;

            solution.weight += weight_change;
            solution.weights[node_i.index] += weight_change;

            let new_fitness = solution.update_cumulative_data(i);
            solution.fitness = new_fitness;

            solution.debug_check_solution_is_valid();

            sub_neighborhood.activate_node(node_i.index);

            has_improved = true;

            if !REMOVING && solution.weight + solution.problem.smallest_weight > solution.problem.capacity_of_knapsack {
                if cfg!(debug_assertions) {
                    for &item in &solution.problem.items {
                        let item = item - 1;
                        if solution.loot[item] == REMOVING {
                            sub_neighborhood.loot.set(item, false);
                        }
                    }
                }

                break 'outer;
            }
        }
    }

    solution.debug_check_solution_is_valid();
    has_improved
}

pub fn fast_local_search_multiple_neighborhoods<const MAX_ITEMS_PER_CITY: usize>(solution: &mut Solution, sub_neighborhood: &mut SubNeighborhood, env: &Env) -> bool
    where LaneCount<MAX_ITEMS_PER_CITY>: SupportedLaneCount {
    let mut local_minimum_neighborhood = -1;

    sub_neighborhood.loot.fill(true);
    if one_flip::<MAX_ITEMS_PER_CITY, true>(solution, sub_neighborhood, env) { local_minimum_neighborhood = 0; }
    if one_flip::<MAX_ITEMS_PER_CITY, false>(solution, sub_neighborhood, env) { local_minimum_neighborhood = 1; }
    sub_neighborhood.debug_assert_no_loot_active(env);

    if fast_local_search_route::<TwoOpt>(solution, sub_neighborhood, env) { local_minimum_neighborhood = 2; }
    if fast_local_search_route::<OneMove>(solution, sub_neighborhood, env) { local_minimum_neighborhood = 3; }

    if local_minimum_neighborhood < 0 {
        return false;
    }

    while env.should_continue() {
        solution.debug_check_solution_is_valid();

        if local_minimum_neighborhood == 0 { break; }
        sub_neighborhood.loot.fill(true);
        if one_flip::<MAX_ITEMS_PER_CITY, true>(solution, sub_neighborhood, env) { local_minimum_neighborhood = 0; }

        if local_minimum_neighborhood == 1 { break; }
        if one_flip::<MAX_ITEMS_PER_CITY, false>(solution, sub_neighborhood, env) { local_minimum_neighborhood = 1; }

        sub_neighborhood.debug_assert_no_loot_active(env);

        if local_minimum_neighborhood == 2 { break; }
        if fast_local_search_route::<TwoOpt>(solution, sub_neighborhood, env) { local_minimum_neighborhood = 2; }

        if local_minimum_neighborhood == 3 { break; }
        if fast_local_search_route::<OneMove>(solution, sub_neighborhood, env) { local_minimum_neighborhood = 3; }
    }

    true
}

fn destroy_route<'a, 'b: 'a>(index: usize, solution: &'a Solution<'b>, paths_removed: &mut Vec<(usize, isize)>) {
    paths_removed.clear();

    let mut cuts = solution.route[index]
        .neighborhood
        .iter()
        .map(|&i| solution.positions[i])
        .collect::<Vec<_>>();

    cuts.sort();

    // Make the cuts
    let mut previous_cut = 0;
    for cut in cuts {
        paths_removed.push((previous_cut, (cut as isize) - 1));
        previous_cut = cut;
    }
    paths_removed.push((previous_cut, (solution.route.len() as isize) - 1));
}

fn repair_route<'a, 'b: 'a>(solution: &'a mut Solution<'b>,
                            route_buffer: &'a mut Vec<&'b Node>,
                            sub_neighborhood: &mut SubNeighborhood,
                            paths_removed: &mut Vec<(usize, isize)>,
                            rng: &mut Xoshiro256PlusPlus) {
    debug_assert!(paths_removed.len() >= 2, "Paths removed should have at least a beginning and end path, found {}.", paths_removed.len());

    route_buffer.clear();

    let Some(mut last_path) = paths_removed.pop() else { return; };
    let mut first_path = paths_removed.remove(0);

    let reverse_zero_path = if rng.gen_bool(0.5) {
        swap(&mut first_path, &mut last_path);
        true
    } else {
        false
    };

    fn copy_to_route<'a, 'b: 'a>(solution: &'a mut Solution<'b>, route_buffer: &'a mut Vec<&'b Node>, iter: impl Iterator<Item=usize>) {
        for i in iter {
            let node = solution.route[i];
            solution.positions[node.index] = route_buffer.len();
            route_buffer.push(node);
        }
    }

    if first_path.1 >= first_path.0 as isize {
        if reverse_zero_path {
            sub_neighborhood.activate_node(solution.route[first_path.0].index);
            copy_to_route(solution, route_buffer, (first_path.0..=first_path.1 as usize).rev());
        } else {
            sub_neighborhood.activate_node(solution.route[first_path.1 as usize].index);
            copy_to_route(solution, route_buffer, first_path.0..=first_path.1 as usize);
        }
    }

    paths_removed.shuffle(rng);

    for path in paths_removed {
        sub_neighborhood.activate_node(solution.route[path.0].index);
        sub_neighborhood.activate_node(solution.route[path.1 as usize].index);

        if rng.gen_bool(0.5) {
            copy_to_route(solution, route_buffer, (path.0..=path.1 as usize).rev());
        } else {
            copy_to_route(solution, route_buffer, path.0..=path.1 as usize);
        }
    }

    if last_path.1 >= last_path.0 as isize {
        if reverse_zero_path {
            sub_neighborhood.activate_node(solution.route[last_path.1 as usize].index);
            copy_to_route(solution, route_buffer, (last_path.0..=last_path.1 as usize).rev());
        } else {
            sub_neighborhood.activate_node(solution.route[last_path.0].index);
            copy_to_route(solution, route_buffer, last_path.0..=last_path.1 as usize);
        }
    }
}

pub fn large_neighborhood_search<'a, 'b: 'a, const MAX_ITEMS_PER_CITY: usize>(best_solution: &'a mut Solution<'b>,
                                                                              sub_neighborhood: &'a mut SubNeighborhood,
                                                                              env: &'a mut Env) -> Result<()>
    where LaneCount<MAX_ITEMS_PER_CITY>: SupportedLaneCount {
    let mut paths_removed = Vec::with_capacity(best_solution.route.len());

    let mut solution = best_solution.clone();
    let mut route_buffer = Vec::with_capacity(solution.route.len());

    let mut iterations_without_improvement = 0;

    env.record(StartHeuristic { fitness: solution.fitness });

    while iterations_without_improvement < env.args.give_up_after && env.should_continue() {
        solution.partial_clone_from(best_solution);

        iterations_without_improvement += 1;

        let iterations_without_improvement_percent = iterations_without_improvement as f64 / env.args.give_up_after as f64;
        let node_cut_amount = (env.args.node_cut_amount as f64 * iterations_without_improvement_percent).ceil() as usize;
        debug_assert!(node_cut_amount >= 1);
        debug_assert!(node_cut_amount <= env.args.give_up_after as usize);

        sub_neighborhood.reset_to(false);

        let cuts = (0..solution.route.len()).choose_multiple(&mut env.rng, node_cut_amount);
        for cut in cuts {
            destroy_route(cut, &solution, &mut paths_removed);
            repair_route(&mut solution, &mut route_buffer, sub_neighborhood, &mut paths_removed, &mut env.rng);

            swap(&mut solution.route, &mut route_buffer);
            solution.debug_check_route_is_valid();
        }
        solution.update_cumulative_data(0);
        solution.update_fitness_with_weights();

        // Didn't generated a different solution, continue
        if solution.fitness == best_solution.fitness {
            env.record_step(node_cut_amount);
            continue;
        }

        // Greedy knapsack and local search
        greedy_picking(&mut solution);
        fast_local_search_multiple_neighborhoods::<MAX_ITEMS_PER_CITY>(&mut solution, sub_neighborhood, env);

        // Reset the improvement count and save the best solution
        if solution.fitness > best_solution.fitness {
            iterations_without_improvement = 0;

            best_solution.clone_from(&solution);
            best_solution.save_to_directory(env)?;

            env.record(ImproveHeuristic { fitness: best_solution.fitness });
        } else {
            env.record_step(node_cut_amount);
        }
    }

    Ok(())
}

pub fn greedy_picking(solution: &mut Solution) {
    let previous_fitness = solution.fitness;

    solution.pick_from_test_power::<false>();
    solution.pick_from_test_power::<true>();

    if solution.fitness > previous_fitness {
        solution.update_cumulative_data(0);
    }
}

pub fn solve<const MAX_ITEMS_PER_CITY: usize>(problem: &Problem, env: &mut Env) -> Result<()>
    where LaneCount<MAX_ITEMS_PER_CITY>: SupportedLaneCount {
    let mut solutions = Vec::with_capacity(env.args.initial_route_amount + 1);
    let mut sub_neighborhood = SubNeighborhood::new(problem);

    'outer: loop {
        solutions.clear();

        let mut best_initial_fitness = f64::NEG_INFINITY;
        let mut iterations_without_improvement = 0;

        while iterations_without_improvement < env.args.initial_route_amount {
            if !env.should_continue() {
                break 'outer;
            }

            iterations_without_improvement += 1;

            let route = problem.lin_kernighan.solve(problem, env)?;
            let mut solution = Solution::from_lk_route(problem, route);

            greedy_picking(&mut solution);
            solution.save_to_directory(env)?;

            sub_neighborhood.reset_to(true);
            fast_local_search_multiple_neighborhoods::<MAX_ITEMS_PER_CITY>(&mut solution, &mut sub_neighborhood, env);
            solution.save_to_directory(env)?;

            if solution.fitness > best_initial_fitness {
                best_initial_fitness = solution.fitness;
                iterations_without_improvement = 0;
            }

            env.record(InitialSolution { fitness: solution.fitness });

            if !env.args.silent {
                println!("finished fls ({:0>2}): got {}, best {}, at {}", solutions.len() + 1, solution.fitness, env.best_fitness, env.timer.elapsed().as_secs_f64());
            }

            solutions.push(solution);
        }

        solutions.sort_by(|a, b| b.fitness.total_cmp(&a.fitness));
        let solutions_count = solutions.len();

        for (i, solution) in solutions.iter_mut().enumerate() {
            if !env.should_continue() {
                break 'outer;
            }

            large_neighborhood_search::<MAX_ITEMS_PER_CITY>(solution, &mut sub_neighborhood, env)?;

            if !env.args.silent {
                println!("finished lns ({:0>2}/{:0>2}): got {}, best {}, at {}",
                         i + 1, solutions_count, solution.fitness, env.best_fitness, env.timer.elapsed().as_secs_f64());
            }
        }
    }

    env.save_log().context("Error saving log.")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use clap::Parser;

    use super::*;

    #[test]
    fn greedy_picking_a280() {
        let args = Args::parse_from(["test", "--silent", "-d", "-s42", "instances/a280_n279_bounded-strongly-corr_01.ttp"]);
        let mut env = Env::from_args(args).unwrap();

        let problem = Problem::new(&env.args.problem_file, &None).unwrap();
        let route = problem.lin_kernighan.solve(&problem, &mut env).unwrap();

        let mut solution = Solution::from_lk_route(&problem, route);
        greedy_picking(&mut solution);

        let expected = "[1, 280, 3, 279, 278, 248, 249, 256, 255, 252, 253, 254, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 140, 141, 142, 143, 146, 147, 148, 149, 139, 138, 137, 267, 268, 136, 135, 269, 270, 134, 133, 18, 17, 16, 271, 272, 273, 274, 275, 276, 277, 4, 5, 6, 7, 9, 8, 10, 11, 12, 13, 15, 14, 24, 23, 25, 22, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 60, 61, 118, 62, 63, 59, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 76, 75, 77, 78, 79, 80, 81, 82, 83, 88, 112, 113, 87, 84, 85, 86, 116, 117, 115, 114, 111, 110, 108, 109, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 174, 173, 172, 171, 170, 169, 168, 167, 166, 165, 164, 163, 162, 161, 175, 160, 159, 158, 157, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 21, 20, 19, 132, 131, 130, 129, 154, 155, 153, 156, 152, 151, 177, 178, 150, 179, 180, 176, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 145, 144, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 251, 250, 247, 244, 245, 246, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 243, 242, 2]
[1, 2, 3, 4, 18, 19, 20, 21, 43, 49, 50, 51, 56, 65, 66, 67, 68, 84, 85, 86, 87, 106, 107, 108, 125, 126, 127, 128, 130, 149, 150, 151, 152, 166, 167, 181, 182, 210, 211, 212, 213, 214, 215, 216, 220, 221, 222, 223, 224, 225, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 249, 250, 257]
";

        assert_eq!(solution.fitness, 18305.98035339103);
        assert_eq!(solution.to_string(), expected);
    }

    #[test]
    fn fast_local_search_a280() {
        let args = Args::parse_from(["test", "--silent", "-d", "-s42", "instances/a280_n279_bounded-strongly-corr_01.ttp"]);
        let mut env = Env::from_args(args).unwrap();

        let problem = Problem::new(&env.args.problem_file, &None).unwrap();
        let route = problem.lin_kernighan.solve(&problem, &mut env).unwrap();

        let mut solution = Solution::from_lk_route(&problem, route);

        solution.update_weights_profit_and_fitness();
        solution.update_cumulative_data(0);

        let mut sub_neighborhood = SubNeighborhood::new(&problem);
        fast_local_search_multiple_neighborhoods::<1>(&mut solution, &mut sub_neighborhood, &env);

        let expected = "[1, 280, 3, 279, 278, 248, 249, 256, 255, 252, 253, 254, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 140, 141, 142, 143, 146, 147, 148, 149, 139, 138, 137, 267, 268, 136, 135, 269, 270, 134, 133, 18, 17, 16, 271, 272, 273, 274, 275, 276, 277, 4, 5, 6, 7, 9, 8, 10, 11, 12, 13, 15, 14, 24, 23, 25, 22, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 60, 61, 118, 62, 63, 59, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 76, 75, 77, 78, 79, 80, 81, 82, 83, 88, 112, 113, 87, 84, 85, 86, 116, 117, 115, 114, 111, 110, 108, 109, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 174, 173, 172, 171, 170, 169, 168, 167, 166, 165, 164, 163, 162, 161, 175, 160, 159, 158, 157, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 21, 20, 19, 132, 131, 130, 129, 154, 155, 153, 156, 152, 151, 177, 178, 150, 179, 180, 176, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 145, 144, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 251, 250, 247, 243, 244, 245, 246, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 2]
[1, 2, 3, 4, 106, 108, 220, 221, 222, 225, 226, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 249]
";

        assert_eq!(solution.fitness, 13384.709756720884);
        assert_eq!(solution.to_string(), expected);
    }

    #[allow(dead_code)] // It's being used, but clippy complains anyway
    fn large_neighborhood_search_a280_n_items<const MAX_ITEMS_PER_CITY: usize>(instance: &'static str, expected_fitness: f64, expected: &'static str) -> Result<()>
        where LaneCount<MAX_ITEMS_PER_CITY>: SupportedLaneCount {
        let args = Args::parse_from(["test", "--silent", "-d", "-g5", "-ll", "-s42", instance]);
        let mut env = Env::from_args(args)?;

        let problem = Problem::new(&env.args.problem_file, &None).unwrap();
        let route = problem.lin_kernighan.solve(&problem, &mut env).unwrap();

        let mut solution = Solution::from_lk_route(&problem, route);
        greedy_picking(&mut solution);
        let mut sub_neighborhood = SubNeighborhood::new(&problem);
        fast_local_search_multiple_neighborhoods::<MAX_ITEMS_PER_CITY>(&mut solution, &mut sub_neighborhood, &env);
        large_neighborhood_search::<MAX_ITEMS_PER_CITY>(&mut solution, &mut sub_neighborhood, &mut env)?;

        assert_eq!(solution.fitness, expected_fitness);
        assert_eq!(solution.to_string(), expected);
        Ok(())
    }

    #[test]
    fn large_neighborhood_search_a280_1_item() {
        let expected = "[1, 280, 3, 279, 278, 248, 249, 256, 255, 252, 253, 254, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 140, 141, 142, 143, 146, 147, 148, 149, 139, 138, 137, 267, 268, 136, 135, 269, 270, 134, 133, 18, 17, 16, 271, 272, 273, 274, 275, 276, 277, 4, 5, 6, 7, 9, 8, 10, 11, 12, 13, 15, 14, 24, 23, 25, 22, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 60, 61, 118, 62, 63, 59, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 76, 75, 77, 78, 79, 80, 81, 82, 83, 88, 112, 113, 87, 84, 85, 86, 116, 117, 115, 114, 111, 110, 108, 109, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 174, 173, 172, 171, 170, 169, 168, 167, 166, 165, 164, 163, 162, 161, 175, 160, 159, 158, 157, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 21, 20, 19, 132, 131, 130, 129, 154, 155, 153, 156, 152, 151, 177, 178, 150, 179, 180, 176, 181, 182, 183, 184, 185, 187, 186, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 145, 144, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 251, 250, 247, 244, 245, 246, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 243, 242, 2]
[1, 2, 3, 4, 18, 19, 20, 21, 43, 49, 50, 51, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 84, 85, 86, 87, 88, 89, 90, 91, 99, 100, 101, 102, 103, 104, 106, 107, 108, 118, 119, 121, 122, 123, 124, 125, 126, 127, 128, 130, 131, 143, 144, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 165, 166, 167, 171, 175, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 249, 250, 255, 256, 257]
";
        large_neighborhood_search_a280_n_items::<1>("instances/a280_n279_bounded-strongly-corr_05.ttp", 55382.1363956821, expected).unwrap();
    }

    #[test]
    fn large_neighborhood_search_a280_3_item() {
        let expected = "[1, 280, 3, 279, 278, 248, 249, 256, 255, 252, 253, 254, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 140, 141, 142, 143, 146, 147, 148, 149, 139, 138, 137, 267, 268, 136, 135, 269, 270, 134, 133, 18, 17, 16, 271, 272, 273, 274, 275, 276, 277, 4, 5, 6, 7, 9, 8, 10, 11, 12, 13, 15, 14, 24, 23, 25, 22, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 60, 61, 118, 62, 63, 59, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 76, 75, 77, 78, 79, 80, 81, 82, 83, 88, 112, 113, 87, 84, 85, 86, 116, 117, 115, 114, 111, 110, 108, 109, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 174, 173, 172, 171, 170, 169, 168, 167, 166, 165, 164, 163, 162, 161, 175, 160, 159, 158, 157, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 21, 20, 19, 132, 131, 130, 129, 154, 155, 153, 156, 152, 151, 177, 178, 150, 179, 180, 176, 181, 182, 183, 184, 185, 187, 186, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 145, 144, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 251, 250, 247, 244, 245, 246, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 243, 242, 2]
[1, 2, 3, 4, 18, 19, 20, 21, 43, 45, 46, 47, 48, 49, 50, 51, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 84, 85, 86, 87, 88, 89, 90, 91, 99, 100, 101, 102, 103, 104, 106, 107, 108, 116, 118, 119, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 143, 144, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 163, 164, 165, 166, 167, 171, 172, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 249, 250, 255, 256, 257, 280, 288, 289, 290, 291, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 329, 330, 331, 332, 340, 341, 342, 364, 365, 366, 367, 368, 369, 370, 371, 372, 384, 385, 386, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 422, 423, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 447, 448, 449, 450, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493, 494, 495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511, 512, 513, 514, 515, 516, 517, 518, 519, 520, 521, 522, 523, 524, 525, 526, 527, 528, 529, 537, 538, 559, 561, 562, 563, 569, 570, 571, 576, 577, 578, 579, 580, 581, 600, 601, 602, 603, 606, 607, 608, 609, 610, 634, 635, 636, 637, 642, 651, 652, 653, 654, 661, 662, 663, 664, 665, 666, 675, 676, 677, 679, 680, 683, 684, 685, 686, 687, 688, 689, 701, 702, 707, 708, 709, 710, 711, 712, 713, 714, 715, 720, 721, 722, 723, 724, 725, 727, 728, 729, 730, 731, 732, 733, 734, 735, 736, 737, 738, 739, 740, 741, 742, 743, 744, 745, 746, 747, 748, 749, 750, 751, 752, 753, 754, 755, 756, 757, 758, 759, 760, 761, 762, 763, 764, 765, 766, 767, 768, 769, 770, 771, 772, 773, 774, 775, 776, 777, 778, 779, 780, 781, 782, 783, 784, 785, 786, 787, 788, 789, 790, 791, 792, 793, 794, 795, 796, 797, 798, 799, 800, 801, 802, 803, 804, 807, 808, 815, 816, 817, 828]
";
        large_neighborhood_search_a280_n_items::<4>("instances/a280_n837_bounded-strongly-corr_05.ttp", 144236.78838383983, expected).unwrap();
    }

    #[test]
    fn large_neighborhood_search_a280_5_item() {
        let expected = "[1, 280, 3, 279, 278, 248, 249, 256, 255, 252, 253, 254, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 140, 141, 142, 143, 146, 147, 148, 149, 139, 138, 137, 267, 268, 136, 135, 269, 270, 134, 133, 18, 17, 16, 271, 272, 273, 274, 275, 276, 277, 4, 5, 6, 7, 9, 8, 10, 11, 12, 13, 15, 14, 24, 23, 25, 22, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 60, 61, 118, 62, 63, 59, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 76, 75, 77, 78, 79, 80, 81, 82, 83, 88, 112, 113, 87, 84, 85, 86, 116, 117, 115, 114, 111, 110, 108, 109, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 174, 173, 172, 171, 170, 169, 168, 167, 166, 165, 164, 163, 162, 161, 175, 160, 159, 158, 157, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 21, 20, 19, 132, 131, 130, 129, 154, 155, 153, 156, 152, 151, 177, 178, 150, 179, 180, 176, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 145, 144, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 251, 250, 247, 244, 245, 246, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 243, 242, 2]
[1, 2, 3, 4, 18, 19, 20, 21, 43, 45, 46, 47, 48, 49, 50, 51, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 84, 85, 86, 87, 88, 89, 90, 91, 99, 100, 101, 102, 103, 104, 106, 107, 108, 116, 118, 119, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 143, 144, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 163, 164, 165, 166, 167, 171, 172, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 249, 250, 255, 256, 257, 280, 288, 289, 290, 291, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 329, 330, 331, 332, 340, 341, 342, 364, 365, 366, 367, 368, 369, 370, 371, 372, 384, 385, 386, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 422, 423, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 447, 448, 449, 450, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493, 494, 495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511, 512, 513, 514, 515, 516, 517, 518, 519, 520, 521, 522, 523, 524, 525, 526, 527, 528, 529, 537, 538, 559, 561, 562, 563, 569, 570, 571, 576, 577, 578, 579, 580, 581, 600, 601, 602, 603, 606, 607, 608, 609, 610, 634, 635, 636, 637, 641, 643, 651, 652, 653, 654, 661, 662, 663, 664, 665, 666, 675, 676, 677, 679, 680, 684, 685, 686, 687, 688, 689, 701, 702, 707, 708, 709, 710, 711, 712, 713, 714, 715, 720, 721, 722, 723, 724, 725, 727, 728, 729, 730, 731, 732, 733, 734, 735, 736, 737, 738, 739, 740, 741, 742, 743, 744, 745, 746, 747, 748, 749, 750, 751, 752, 753, 754, 755, 756, 757, 758, 759, 760, 761, 762, 763, 764, 765, 766, 767, 768, 769, 770, 771, 772, 773, 774, 775, 776, 777, 778, 779, 780, 781, 782, 783, 784, 785, 786, 787, 788, 789, 790, 791, 792, 793, 794, 795, 796, 797, 798, 799, 800, 801, 802, 803, 804, 807, 808, 815, 816, 817, 828, 838, 848, 853, 854, 855, 856, 866, 897, 898, 901, 902, 903, 904, 916, 917, 926, 927, 928, 929, 930, 931, 932, 933, 934, 935, 942, 943, 944, 945, 951, 952, 953, 954, 955, 956, 957, 958, 959, 960, 961, 962, 963, 964, 965, 966, 967, 968, 980, 981, 984, 985, 986, 987, 988, 989, 990, 991, 992, 994, 995, 996, 997, 1001, 1002, 1003, 1004, 1012, 1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023, 1024, 1025, 1026, 1027, 1028, 1029, 1030, 1031, 1032, 1033, 1034, 1035, 1036, 1037, 1038, 1039, 1040, 1041, 1042, 1043, 1044, 1045, 1046, 1047, 1048, 1049, 1050, 1051, 1052, 1053, 1054, 1055, 1056, 1057, 1058, 1059, 1060, 1061, 1062, 1063, 1064, 1065, 1066, 1067, 1068, 1069, 1070, 1071, 1072, 1073, 1074, 1075, 1076, 1077, 1078, 1079, 1080, 1081, 1082, 1083, 1086, 1087, 1099, 1100, 1101, 1102, 1109, 1110, 1117, 1139, 1140, 1141, 1142, 1143, 1144, 1145, 1146, 1147, 1148, 1149, 1153, 1154, 1155, 1156, 1157, 1158, 1162, 1163, 1175, 1176, 1177, 1178, 1179, 1180, 1181, 1182, 1183, 1184, 1185, 1186, 1187, 1188, 1189, 1193, 1194, 1195, 1196, 1197, 1198, 1206, 1207, 1215, 1216, 1217, 1218, 1219, 1220, 1221, 1222, 1223, 1224, 1233, 1234, 1235, 1242, 1243, 1244, 1245, 1246, 1247, 1259, 1260, 1263, 1264, 1265, 1266, 1267, 1268, 1269, 1270, 1271, 1272, 1273, 1274, 1275, 1276, 1277, 1278, 1279, 1280, 1281, 1284, 1285, 1286, 1287, 1291, 1292, 1293, 1294, 1295, 1296, 1297, 1298, 1299, 1300, 1301, 1302, 1303, 1304, 1305, 1306, 1307, 1308, 1309, 1310, 1311, 1312, 1313, 1314, 1315, 1316, 1317, 1318, 1319, 1320, 1321, 1322, 1323, 1324, 1325, 1326, 1327, 1328, 1329, 1330, 1331, 1332, 1333, 1334, 1335, 1336, 1337, 1338, 1339, 1340, 1341, 1342, 1343, 1344, 1345, 1346, 1347, 1348, 1349, 1350, 1351, 1352, 1353, 1354, 1355, 1356, 1357, 1358, 1359, 1360, 1361, 1362, 1365, 1366, 1372, 1373, 1374, 1389, 1390, 1391]
";
        large_neighborhood_search_a280_n_items::<8>("instances/a280_n1395_bounded-strongly-corr_05.ttp", 246989.51153450977, expected).unwrap();
    }

    #[test]
    fn large_neighborhood_search_a280_10_item() {
        let expected = "[1, 280, 3, 279, 278, 248, 249, 256, 255, 252, 253, 254, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 140, 141, 142, 143, 146, 147, 148, 149, 139, 138, 137, 267, 268, 136, 135, 269, 270, 134, 133, 18, 17, 16, 271, 272, 273, 274, 275, 276, 277, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 14, 24, 23, 25, 22, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 60, 61, 118, 62, 63, 59, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 76, 75, 77, 78, 79, 80, 81, 82, 83, 88, 112, 113, 87, 84, 85, 86, 116, 117, 115, 114, 111, 110, 108, 109, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 174, 173, 172, 171, 170, 169, 168, 167, 166, 165, 164, 163, 162, 161, 175, 160, 159, 158, 157, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 21, 20, 19, 132, 131, 130, 129, 154, 155, 153, 156, 152, 151, 177, 178, 150, 179, 180, 176, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 145, 144, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 251, 250, 247, 244, 245, 246, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 243, 242, 2]
[1, 2, 3, 4, 18, 19, 20, 21, 43, 45, 46, 47, 48, 49, 50, 51, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 84, 85, 86, 87, 88, 89, 90, 91, 99, 100, 101, 102, 103, 104, 106, 107, 108, 116, 118, 119, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 143, 144, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 163, 164, 165, 166, 167, 171, 172, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 249, 250, 255, 256, 257, 280, 288, 289, 290, 291, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 329, 330, 331, 332, 340, 341, 342, 364, 365, 366, 367, 368, 369, 384, 385, 386, 395, 396, 397, 398, 399, 400, 401, 402, 403, 405, 406, 407, 408, 409, 410, 411, 412, 413, 422, 423, 428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 447, 448, 449, 450, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493, 494, 495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511, 512, 513, 514, 515, 516, 517, 518, 519, 520, 521, 522, 523, 524, 525, 526, 527, 528, 529, 537, 538, 559, 561, 562, 563, 569, 570, 571, 576, 577, 578, 579, 580, 581, 600, 601, 602, 603, 606, 607, 608, 609, 610, 634, 635, 636, 637, 651, 652, 653, 654, 661, 662, 663, 664, 665, 666, 675, 676, 677, 679, 680, 684, 685, 686, 687, 688, 689, 701, 702, 707, 708, 709, 710, 711, 712, 713, 714, 715, 720, 721, 722, 723, 724, 725, 727, 728, 729, 730, 731, 732, 733, 734, 735, 736, 737, 738, 739, 740, 741, 742, 743, 744, 745, 746, 747, 748, 749, 750, 751, 752, 753, 754, 755, 756, 757, 758, 759, 760, 761, 762, 763, 764, 765, 766, 767, 768, 769, 770, 771, 772, 773, 774, 775, 776, 777, 778, 779, 780, 781, 782, 783, 784, 785, 786, 787, 788, 789, 790, 791, 792, 793, 794, 795, 796, 797, 798, 799, 800, 801, 802, 803, 804, 807, 808, 815, 816, 817, 828, 838, 848, 853, 854, 855, 866, 897, 898, 901, 902, 903, 904, 916, 917, 926, 927, 928, 929, 930, 931, 932, 933, 934, 935, 942, 943, 944, 945, 951, 952, 953, 954, 955, 956, 957, 958, 959, 960, 961, 962, 963, 964, 965, 966, 967, 980, 981, 984, 985, 986, 987, 988, 989, 990, 991, 992, 994, 995, 996, 997, 1001, 1002, 1003, 1004, 1012, 1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023, 1024, 1025, 1026, 1027, 1028, 1029, 1030, 1031, 1032, 1033, 1034, 1035, 1036, 1037, 1038, 1039, 1040, 1041, 1042, 1043, 1044, 1045, 1046, 1047, 1048, 1049, 1050, 1051, 1052, 1053, 1054, 1055, 1056, 1057, 1058, 1059, 1060, 1061, 1062, 1063, 1064, 1065, 1066, 1067, 1068, 1069, 1070, 1071, 1072, 1073, 1074, 1075, 1076, 1077, 1078, 1079, 1080, 1081, 1082, 1083, 1086, 1087, 1099, 1100, 1101, 1102, 1109, 1110, 1117, 1139, 1140, 1141, 1142, 1143, 1144, 1145, 1146, 1147, 1148, 1149, 1153, 1154, 1155, 1156, 1157, 1158, 1162, 1163, 1175, 1176, 1177, 1178, 1179, 1180, 1181, 1182, 1183, 1184, 1185, 1186, 1187, 1188, 1189, 1193, 1194, 1195, 1196, 1197, 1198, 1206, 1207, 1215, 1216, 1217, 1218, 1219, 1220, 1221, 1222, 1223, 1224, 1233, 1234, 1235, 1242, 1243, 1244, 1245, 1246, 1259, 1260, 1263, 1264, 1265, 1266, 1267, 1268, 1269, 1270, 1271, 1272, 1273, 1274, 1275, 1276, 1277, 1278, 1279, 1280, 1281, 1284, 1285, 1286, 1287, 1291, 1292, 1293, 1294, 1295, 1296, 1297, 1298, 1299, 1300, 1301, 1302, 1303, 1304, 1305, 1306, 1307, 1308, 1309, 1310, 1311, 1312, 1313, 1314, 1315, 1316, 1317, 1318, 1319, 1320, 1321, 1322, 1323, 1324, 1325, 1326, 1327, 1328, 1329, 1330, 1331, 1332, 1333, 1334, 1335, 1336, 1337, 1338, 1339, 1340, 1341, 1342, 1343, 1344, 1345, 1346, 1347, 1348, 1349, 1350, 1351, 1352, 1353, 1354, 1355, 1356, 1357, 1358, 1359, 1360, 1361, 1362, 1365, 1366, 1372, 1373, 1374, 1389, 1390, 1391, 1396, 1411, 1413, 1414, 1415, 1419, 1420, 1421, 1426, 1427, 1428, 1429, 1430, 1435, 1436, 1437, 1438, 1439, 1440, 1441, 1442, 1446, 1447, 1448, 1456, 1457, 1458, 1459, 1460, 1461, 1462, 1463, 1464, 1465, 1466, 1467, 1468, 1469, 1470, 1471, 1472, 1482, 1483, 1484, 1485, 1486, 1487, 1488, 1491, 1496, 1497, 1499, 1500, 1501, 1502, 1503, 1504, 1505, 1508, 1509, 1510, 1513, 1514, 1515, 1516, 1517, 1518, 1519, 1520, 1521, 1524, 1526, 1538, 1539, 1544, 1545, 1546, 1547, 1548, 1549, 1550, 1559, 1560, 1561, 1562, 1563, 1564, 1565, 1566, 1567, 1568, 1570, 1571, 1572, 1573, 1574, 1575, 1576, 1577, 1578, 1579, 1580, 1581, 1582, 1583, 1584, 1585, 1586, 1587, 1588, 1589, 1590, 1591, 1592, 1593, 1594, 1595, 1596, 1597, 1598, 1599, 1600, 1601, 1602, 1603, 1604, 1605, 1606, 1607, 1608, 1609, 1610, 1611, 1612, 1613, 1614, 1615, 1616, 1617, 1618, 1619, 1620, 1621, 1622, 1623, 1624, 1625, 1626, 1627, 1628, 1629, 1630, 1631, 1632, 1633, 1634, 1635, 1636, 1637, 1638, 1639, 1640, 1641, 1644, 1645, 1659, 1660, 1661, 1662, 1663, 1664, 1665, 1666, 1675, 1682, 1683, 1692, 1693, 1694, 1695, 1696, 1697, 1698, 1699, 1702, 1703, 1704, 1705, 1706, 1707, 1708, 1709, 1710, 1711, 1712, 1715, 1716, 1717, 1718, 1719, 1720, 1733, 1734, 1735, 1753, 1759, 1760, 1761, 1762, 1774, 1775, 1776, 1777, 1778, 1779, 1780, 1781, 1782, 1783, 1784, 1785, 1786, 1787, 1788, 1789, 1790, 1792, 1793, 1796, 1797, 1798, 1799, 1802, 1803, 1804, 1805, 1817, 1818, 1821, 1822, 1823, 1824, 1825, 1826, 1827, 1828, 1829, 1833, 1834, 1835, 1836, 1837, 1838, 1839, 1840, 1841, 1842, 1843, 1844, 1849, 1850, 1851, 1852, 1853, 1854, 1855, 1856, 1857, 1858, 1859, 1860, 1861, 1862, 1863, 1864, 1865, 1866, 1867, 1868, 1869, 1870, 1871, 1872, 1873, 1874, 1875, 1876, 1877, 1878, 1879, 1880, 1881, 1882, 1883, 1884, 1885, 1886, 1887, 1888, 1889, 1890, 1891, 1892, 1893, 1894, 1895, 1896, 1897, 1898, 1899, 1900, 1901, 1902, 1903, 1904, 1905, 1906, 1907, 1908, 1909, 1910, 1911, 1912, 1913, 1914, 1915, 1916, 1917, 1918, 1919, 1920, 1923, 1924, 1930, 1936, 1937, 1942, 1943, 1944, 1954, 1955, 1956, 1957, 1958, 1971, 1973, 1977, 1978, 1979, 2003, 2004, 2005, 2006, 2017, 2018, 2019, 2027, 2028, 2029, 2030, 2031, 2032, 2033, 2034, 2035, 2036, 2037, 2038, 2039, 2040, 2047, 2048, 2049, 2052, 2053, 2054, 2055, 2056, 2057, 2058, 2059, 2066, 2067, 2068, 2069, 2070, 2071, 2072, 2073, 2074, 2075, 2076, 2077, 2078, 2079, 2080, 2081, 2082, 2083, 2084, 2096, 2097, 2102, 2103, 2104, 2105, 2106, 2107, 2108, 2109, 2110, 2111, 2112, 2126, 2127, 2128, 2129, 2130, 2131, 2132, 2133, 2134, 2135, 2136, 2137, 2138, 2139, 2140, 2141, 2142, 2143, 2144, 2145, 2146, 2147, 2148, 2149, 2150, 2151, 2152, 2153, 2154, 2155, 2156, 2157, 2158, 2159, 2160, 2161, 2162, 2163, 2164, 2165, 2166, 2167, 2168, 2169, 2170, 2171, 2172, 2173, 2174, 2175, 2176, 2177, 2178, 2179, 2180, 2181, 2182, 2183, 2184, 2185, 2186, 2187, 2188, 2189, 2190, 2191, 2192, 2193, 2194, 2195, 2196, 2197, 2198, 2199, 2202, 2203, 2206, 2207, 2208, 2209, 2217, 2218, 2219, 2233, 2249, 2250, 2251, 2252, 2269, 2270, 2271, 2272, 2273, 2274, 2275, 2279, 2280, 2281, 2283, 2284, 2285, 2286, 2290, 2291, 2292, 2297, 2298, 2305, 2306, 2307, 2318, 2319, 2320, 2321, 2329, 2330, 2331, 2332, 2333, 2334, 2338, 2339, 2340, 2350, 2351, 2352, 2357, 2358, 2359, 2360, 2361, 2362, 2363, 2366, 2367, 2375, 2376, 2381, 2382, 2383, 2384, 2385, 2386, 2387, 2392, 2393, 2394, 2395, 2396, 2397, 2401, 2402, 2403, 2404, 2405, 2406, 2407, 2408, 2409, 2410, 2411, 2412, 2413, 2414, 2415, 2416, 2417, 2418, 2419, 2420, 2421, 2422, 2423, 2424, 2425, 2426, 2427, 2428, 2429, 2430, 2431, 2432, 2433, 2434, 2435, 2436, 2437, 2438, 2439, 2440, 2441, 2442, 2443, 2444, 2445, 2446, 2447, 2448, 2449, 2450, 2451, 2452, 2453, 2454, 2455, 2456, 2457, 2458, 2459, 2460, 2461, 2462, 2463, 2464, 2465, 2466, 2467, 2468, 2469, 2470, 2471, 2472, 2473, 2474, 2475, 2476, 2477, 2478, 2479, 2480, 2481, 2482, 2494, 2495, 2496, 2512, 2529, 2530, 2531, 2532, 2533, 2534, 2535, 2536, 2537, 2538, 2551, 2552, 2553, 2554, 2555, 2558, 2559, 2560, 2565, 2566, 2567, 2571, 2572, 2573, 2574, 2584, 2585, 2586, 2590, 2591, 2599, 2608, 2609, 2610, 2611, 2612, 2613, 2614, 2626, 2627, 2628, 2629, 2630, 2631, 2632, 2633, 2634, 2635, 2636, 2654, 2655, 2660, 2661, 2662, 2663, 2664, 2665, 2666, 2669, 2670, 2671, 2672, 2673, 2677, 2678, 2679, 2680, 2681, 2682, 2683, 2684, 2686, 2687, 2688, 2689, 2690, 2691, 2692, 2693, 2694, 2695, 2696, 2697, 2698, 2699, 2700, 2701, 2702, 2703, 2704, 2705, 2706, 2707, 2708, 2709, 2710, 2711, 2712, 2713, 2714, 2715, 2716, 2717, 2718, 2719, 2720, 2721, 2722, 2723, 2724, 2725, 2726, 2727, 2728, 2729, 2730, 2731, 2732, 2733, 2734, 2735, 2736, 2737, 2738, 2739, 2740, 2741, 2742, 2743, 2744, 2745, 2746, 2747, 2748, 2749, 2750, 2751, 2752, 2753, 2754, 2755, 2756, 2757, 2760, 2761, 2762, 2763, 2764, 2768, 2769, 2784, 2785, 2786, 2787, 2788, 2789, 2790]
";
        large_neighborhood_search_a280_n_items::<16>("instances/a280_n2790_bounded-strongly-corr_05.ttp", 471227.82511186396, expected).unwrap();
    }
}
