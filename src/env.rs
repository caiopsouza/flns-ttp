use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::Context;
use anyhow::Result;
use clap::Parser;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use uuid::Uuid;
use crate::env::RecordType::StepHeuristic;

#[derive(Copy, Clone)]
pub enum RecordType {
    InitialSolution { fitness: f64 },
    StartHeuristic { fitness: f64 },
    StepHeuristic { amount: u64, cuts: usize },
    ImproveHeuristic { fitness: f64 },
}

#[derive(Clone)]
pub struct Record {
    pub(crate) rtype: RecordType,
    pub(crate) timestamp: f64,
}

#[derive(Parser, Debug, Clone)]
pub struct Args {
    /// Instance input file
    #[arg(value_hint = clap::ValueHint::FilePath)]
    pub problem_file: PathBuf,

    /// Check if a solution is valid, display its fitness and finishes
    #[arg(short, long, value_hint = clap::ValueHint::DirPath)]
    pub check_solution: Option<PathBuf>,

    /// Check if the extra info of the solution is valid [default: false]
    #[arg(short = 'x', long, default_value_t = false)]
    pub extra_info_check: bool,

    /// Output directory to write the best solution found so far
    #[arg(short, long, value_hint = clap::ValueHint::DirPath, default_value = ".")]
    pub output_dir: PathBuf,

    /// Don't save best solution found [default: false]
    #[arg(short, long, default_value_t = false)]
    pub dont_save_solution: bool,

    /// Print nothing to the screen [default: false]
    #[arg(long, default_value_t = false)]
    pub silent: bool,

    /// Save log. Has two versions: only parameters or execution data
    #[arg(short = 'l', long, default_value_t = 0, action = clap::ArgAction::Count)]
    pub save_log: u8,

    /// Vendor location [default: "./vendor/]
    #[arg(short, long, value_hint = clap::ValueHint::DirPath)]
    pub vendor: Option<PathBuf>,

    /// Stops after running a local search that exceeds this time
    #[arg(short, long, default_value_t = Duration::from_secs(10 * 60).into())]
    pub timeout: humantime::Duration,

    /// Seed for the PRNG
    #[arg(short, long, default_value_t = 1)]
    pub seed: u64,

    /// Amount of initial solutions
    #[arg(short = 'r', long, default_value_t = 5)]
    pub initial_route_amount: usize,

    /// Amount of nodes to cut the route
    #[arg(short, long, default_value_t = 5)]
    pub node_cut_amount: usize,

    /// Number of iterations without improvement to give up
    #[arg(short, long, default_value_t = 5_000)]
    pub give_up_after: u64,
}

#[derive(Clone)]
pub struct Env {
    pub args: Args,
    pub rng: Xoshiro256PlusPlus,
    pub timer: Instant,
    pub timeout: Duration,
    pub output_file: PathBuf,
    pub best_fitness: f64,
    pub time_at_best_fitness: f64,
    should_record: bool,
    pub records: Vec<Record>,
}

impl Env {
    pub fn from_args(args: Args) -> Result<Self> {
        let output_file_name = args.problem_file
            .file_name().context("Couldn't read the input file's name.")?
            .to_str().context("Couldn't convert the input file's name.")?;

        let unique_id = Uuid::new_v4().to_string();
        let mut output_file: PathBuf = args.output_dir.clone();
        output_file.push(format!("{output_file_name}.{unique_id}.flns"));

        Ok(Env {
            rng: Xoshiro256PlusPlus::seed_from_u64(args.seed),
            timer: Instant::now(),
            timeout: args.timeout.into(),
            output_file,
            best_fitness: f64::NEG_INFINITY,
            time_at_best_fitness: f64::NEG_INFINITY,
            records: vec![],
            should_record: args.save_log > 1,
            args,
        })
    }

    pub fn new() -> Result<Self> {
        Self::from_args(Args::parse())
    }

    #[inline]
    pub(crate) fn should_continue(&self) -> bool {
        self.timer.elapsed() < self.timeout
    }

    #[inline]
    pub(crate) fn record(&mut self, rtype: RecordType) {
        if !self.should_record { return; }

        self.records.push(Record {
            rtype,
            timestamp: self.timer.elapsed().as_secs_f64(),
        });
    }

    #[inline]
    pub(crate) fn record_step(&mut self, cuts: usize) {
        if !self.should_record { return; }

        let now = self.timer.elapsed().as_secs_f64();

        match self.records.last_mut() {
            Some(Record {
                     rtype: StepHeuristic {
                         amount,
                         cuts: cuts_at_top
                     },
                     timestamp
                 }) if *cuts_at_top == cuts => {
                *amount += 1;
                *timestamp = now;
            }
            _ => {
                self.records.push(Record {
                    rtype: StepHeuristic { amount: 1, cuts },
                    timestamp: now,
                });
            }
        }
    }

    pub(crate) fn save_log(&self) -> Result<()> {
        if self.args.save_log == 0 {
            return Ok(());
        }

        let write_error_context = || format!("Couldn't write to file {}.", self.output_file.display());

        let mut log_file = self.output_file.clone();
        log_file.set_extension("flns.log");
        let mut log_file = File::create(log_file).with_context(write_error_context)?;

        writeln!(log_file, "time: {}\nfitness: {}\ntimeout: {}\nseed: {}\ninitial_route_amount: {}\nnode_cut_amount: {}\ngive_up_after: {}",
                 self.time_at_best_fitness, self.best_fitness, self.args.timeout, self.args.seed, self.args.initial_route_amount, self.args.node_cut_amount, self.args.give_up_after)
            .with_context(write_error_context)?;

        if self.args.save_log == 1 {
            return Ok(());
        }

        for record in &self.records {
            let (rtype, rinfo) = match record.rtype {
                RecordType::InitialSolution { fitness } => ("fast", fitness.to_string()),
                RecordType::StartHeuristic { fitness } => ("heur", fitness.to_string()),
                RecordType::StepHeuristic { amount, cuts } => ("step", format!("{amount}\t{cuts}")),
                RecordType::ImproveHeuristic { fitness } => ("impr", fitness.to_string()),
            };

            writeln!(log_file, "{rtype} {:<13} {rinfo}", record.timestamp).with_context(write_error_context)?;
        }

        Ok(())
    }
}
