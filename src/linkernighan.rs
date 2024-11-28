use std::env::temp_dir;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result};
use rand::RngCore;
use uuid::Uuid;

use crate::{Env, Problem};
use crate::problem::Node;

pub struct LinKernighan {
    work_dir: PathBuf,
    vendor_dir: PathBuf,
    input_file: String,
}

impl LinKernighan {
    fn temp_file_path(work_dir: &Path, extension: &str) -> PathBuf {
        let filename = Uuid::new_v4().to_string();
        work_dir.join(PathBuf::from(format!("{filename}.{extension}")))
    }

    pub(crate) fn new(nodes: &Vec<Node>, vendor_dir: &Option<PathBuf>) -> Result<Self> {
        let vendor_dir = vendor_dir.clone().unwrap_or_else(|| PathBuf::from("./vendor/"));
        let vendor_dir = vendor_dir.canonicalize()
            .context(format!("Couldn't find the `{}` directory.\nFollow the instructions on the README.md to create the requirements.", vendor_dir.display()))?;

        let work_dir = temp_dir();

        let tsp_file_path = Self::temp_file_path(&work_dir, "tsp");
        let tsp_file_name = tsp_file_path.display().to_string();

        let mut tsp_file = File::create(tsp_file_path).context("Couldn't create TSP instance file.")?;
        writeln!(tsp_file,
                 "NAME: TTP initial solution\nTYPE: TSP\nDIMENSION: {}\nEDGE_WEIGHT_TYPE: CEIL_2D\nNODE_COORD_SECTION",
                 nodes.len()
        ).context("Couldn't write TSP instance file.")?;

        for node in nodes {
            writeln!(tsp_file, "{} {} {}", node.index + 1, node.x, node.y)
                .context("Couldn't write TSP instance file.")?;
        }

        Ok(Self {
            work_dir,
            vendor_dir,
            input_file: tsp_file_name,
        })
    }

    pub fn solve<'a>(&self, problem: &'a Problem, env: &mut Env) -> Result<Vec<&'a Node>> {
        let program_name = "linkern";
        let mut program = self.vendor_dir.clone();
        program.push(program_name);

        let seed = &env.rng.next_u32().to_string();

        let tour_file_path = Self::temp_file_path(&self.work_dir, "tour");
        let tour_file_name = tour_file_path.display().to_string();

        let mut command = Command::new(program);
        let command = command.current_dir(&self.work_dir);
        command.args(["-s", seed, "-Q", "-o", &tour_file_name, &self.input_file]);

        command.output()
            .context(format!("Couldn't execute the `{program_name}` program in the `{}` directory.\nFollow the instructions on the README.md to create the requirements.",
                             self.vendor_dir.display()))?;

        let route = Self::read_route(problem, tour_file_path)?;
        Ok(route)
    }

    fn read_route(problem: &Problem, tour_file_path: PathBuf) -> Result<Vec<&Node>> {
        let tour_file = File::open(&tour_file_path)
            .context(format!("Couldn't read result file `{}`.", tour_file_path.display()))?;

        let mut reader = BufReader::new(tour_file);
        let mut line = String::with_capacity(200);

        let mut line_counter = 0;

        reader.read_line(&mut line).context("Couldn't read `linkern` output file")?;
        line.clear();

        let mut route = Vec::with_capacity(problem.nodes.len());
        while let Ok(bytes_read) = reader.read_line(&mut line) {
            line_counter += 1;

            if bytes_read == 0 {
                break;
            }

            if let Some(node) = line.split_whitespace().next() {
                let node = node
                    .parse::<usize>()
                    .context(format!("Error parsing linkern output file: Couldn't read index `{line}` at line {line_counter}."))?;

                route.push(&problem.nodes[node]);
            }

            line.clear();
        }
        route.push(&problem.nodes[0]);

        Ok(route)
    }

    pub(crate) fn triangulate(&self, node_count: usize) -> Result<Vec<Vec<usize>>> {
        let program_name = "edgegen";
        let mut program = self.vendor_dir.clone();
        program.push(program_name);

        let edges_file_path = Self::temp_file_path(&self.work_dir, "edges");
        let edges_file_name = edges_file_path.display().to_string();

        let mut command = Command::new(program);
        let command = command.current_dir(&self.work_dir);
        command.args(["-d", "-o", &edges_file_name, &self.input_file]);

        command.output()
            .context(format!("Couldn't execute the `{program_name}` program in the `{}` directory.\nFollow the instructions on the README.md to create the requirements.",
                             self.vendor_dir.display()))?;

        let edges = Self::read_edges(edges_file_path, node_count)?;
        Ok(edges)
    }

    fn read_edges(edges_file_path: PathBuf, node_count: usize) -> Result<Vec<Vec<usize>>> {
        let edges_file = File::open(&edges_file_path)
            .context(format!("Couldn't read result file `{}`.", edges_file_path.display()))?;

        let mut reader = BufReader::new(edges_file);
        let mut line = String::with_capacity(200);

        reader.read_line(&mut line).context("Couldn't read `linkern` output file")?;
        line.clear();

        let mut edges = vec![vec![]; node_count];
        while let Ok(bytes_read) = reader.read_line(&mut line) {
            if bytes_read == 0 {
                break;
            }

            let mut nodes = line.split_whitespace();

            let Some(node) = nodes.next() else { continue; };
            let Some(neighbor) = nodes.next() else { continue; };

            let Some(node) = node.parse::<usize>().ok() else { continue; };
            let Some(neighbor) = neighbor.parse::<usize>().ok() else { continue; };

            edges[node].push(neighbor);
            edges[neighbor].push(node);

            line.clear();
        }

        Ok(edges)
    }
}
