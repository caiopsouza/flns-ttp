#![feature(portable_simd)]

use std::simd::{LaneCount, SupportedLaneCount};
use clap::Parser;

use criterion::{black_box, Criterion, criterion_group, criterion_main};

use flns::{Args, Env, fast_local_search_multiple_neighborhoods, greedy_picking, large_neighborhood_search, Problem, Solution, SubNeighborhood};

fn benchmark_lns<const MAX_ITEMS_PER_CITY: usize>(c: &mut Criterion, items_per_city: usize, instance: &'static str)
    where LaneCount<MAX_ITEMS_PER_CITY>: SupportedLaneCount {
    let args = Args::parse_from(["test", "--silent", "-d", "-g5", "-s42", instance]);
    let mut env = Env::from_args(args).unwrap();
    let problem = Problem::new(&env.args.problem_file, &None).unwrap();
    let route = problem.lin_kernighan.solve(&problem, &mut env).unwrap();
    let mut solution_items = Solution::from_lk_route(&problem, route);
    greedy_picking(&mut solution_items);
    let mut sub_neighborhood = SubNeighborhood::new(&problem);
    fast_local_search_multiple_neighborhoods::<MAX_ITEMS_PER_CITY>(&mut solution_items, &mut sub_neighborhood, &env);

    c.bench_function(&format!("large neighborhood search {items_per_city} items"), |b| b.iter(|| {
        let solution_items = solution_items.clone();
        let mut sub_neighborhood = SubNeighborhood::new(&problem);

        large_neighborhood_search::<MAX_ITEMS_PER_CITY>(
            black_box(&mut solution_items.clone()),
            black_box(&mut sub_neighborhood),
            black_box(&mut env.clone()),
            &mut vec![]).unwrap();
    }));
}

fn criterion_benchmark(c: &mut Criterion) {
    let args = Args::parse_from(["test", "--silent", "-d", "-g3", "-s64", "instances/a280_n279_bounded-strongly-corr_01.ttp"]);
    let mut env = Env::from_args(args).unwrap();
    let problem = Problem::new(&env.args.problem_file, &None).unwrap();
    let route = problem.lin_kernighan.solve(&problem, &mut env).unwrap();
    let solution = Solution::from_lk_route(&problem, route);

    let mut solution_greedy_picked = solution.clone();
    greedy_picking(&mut solution_greedy_picked);

    c.bench_function("greedy picking", |b| b.iter(|| {
        greedy_picking(black_box(&mut solution.clone()));
    }));

    c.bench_function("fast local search", |b| b.iter(|| {
        let mut sub_neighborhood = SubNeighborhood::new(&problem);
        fast_local_search_multiple_neighborhoods::<1>(black_box(&mut solution_greedy_picked.clone()), black_box(&mut sub_neighborhood), black_box(&env));
    }));

    benchmark_lns::<1>(c, 1, "instances/a280_n279_bounded-strongly-corr_05.ttp");
    benchmark_lns::<4>(c, 3, "instances/a280_n837_bounded-strongly-corr_05.ttp");
    benchmark_lns::<8>(c, 5, "instances/a280_n1395_bounded-strongly-corr_05.ttp");
    benchmark_lns::<16>(c, 10, "instances/a280_n2790_bounded-strongly-corr_05.ttp");
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
