use anyhow::{bail, Context, ensure, Result};

use flns::{Env, Problem, Solution, solve};

fn main() -> Result<()> {
    let mut env = Env::new()?;
    let problem = Problem::new(&env.args.problem_file, &env.args.vendor)?;

    if let Some(check_solution) = env.args.check_solution {
        let (solution, extra_fitness) = Solution::from_file(&problem, &check_solution, env.args.extra_info_check)
            .context("Error checking solution.")?;

        if env.args.extra_info_check {
            let Some(extra_fitness) = extra_fitness else { bail!("Couldn't read fitness from log file.") };
            ensure!(extra_fitness == solution.fitness, "Wrong fitness. Expected {extra_fitness}, actual is {}", solution.fitness);
        }

        let log_file_opinion = if env.args.extra_info_check { ", log file agrees" } else { "" };
        println!("Solution is valid.\nFitness is {}{log_file_opinion}.", solution.fitness);
        println!("Finished in {} seconds.", env.timer.elapsed().as_secs_f64());
        return Ok(());
    }

    match problem.max_items_per_city.next_power_of_two() {
        0 => unreachable!("`next_power_of_two` is always positive."),
        1 => solve::<1>(&problem, &mut env),
        2 => solve::<2>(&problem, &mut env),
        4 => solve::<4>(&problem, &mut env),
        8 => solve::<8>(&problem, &mut env),
        16 => solve::<16>(&problem, &mut env),
        32 => solve::<32>(&problem, &mut env),
        _ => {
            if problem.max_items_per_city > 64 {
                println!("Warning: This program consider at most 64 selected or unselected items per node when performing the local search.");
                println!("It might generate sub-par solutions for this instance.");
            }

            solve::<64>(&problem, &mut env)
        }
    }
}
