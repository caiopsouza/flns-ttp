use std::any::type_name;
use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::str::{FromStr, SplitWhitespace};

use anyhow::{Context, Result};
use rustc_hash::FxHashMap;
use crate::linkernighan::LinKernighan;

#[derive(Clone, Default)]
pub(crate) struct Item {
    pub(crate) index: usize,
    pub(crate) profit: f64,
    pub(crate) weight: f64,
}

pub struct Node {
    pub(crate) index: usize,
    pub(crate) x: f64,
    pub(crate) y: f64,
    pub(crate) items: Vec<Item>,
    pub(crate) neighborhood: Vec<usize>,
    pub(crate) adjacency: Vec<usize>,
}

impl Node {
    #[inline]
    pub(crate) fn distance(&self, other: &Self) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;

        (dx * dx + dy * dy).sqrt().ceil()
    }
}

pub struct Problem {
    pub(crate) nodes: Vec<Node>,
    pub(crate) items: Vec<usize>,
    pub(crate) smallest_weight: f64,
    pub max_items_per_city: usize,
    pub(crate) capacity_of_knapsack: f64,
    pub(crate) min_speed: f64,
    pub(crate) max_speed: f64,
    pub(crate) renting_ratio: f64,
    pub lin_kernighan: LinKernighan,
}

impl Problem {
    fn get_header_value<T: FromStr>(header: &FxHashMap<String, String>, key: &str) -> Result<T>
        where <T as FromStr>::Err: Error + Send + Sync + 'static {
        let value = header.get(key)
            .context(format!("Couldn't find `{key}` while reading file's header."))?;

        value.parse::<T>()
            .context(format!("Couldn't parse `{key}`'s value `{value}` as `{}` while reading file's header.", type_name::<T>()))
    }

    fn get_section_value<T: FromStr>(split: &mut SplitWhitespace, key: &str, section: &str, line_counter: usize) -> Result<T>
        where <T as FromStr>::Err: Error + Send + Sync + 'static {
        let value = split.next()
            .context(format!("Couldn't read `{key}` of {section} at line {line_counter}."))?;

        value.parse()
            .context(format!("Couldn't parse `{key}`'s value `{value}` of {section} as `{}` at line {line_counter}.", type_name::<T>()))
    }

    pub fn new(filename: &PathBuf, vendor_dir: &Option<PathBuf>) -> Result<Self> {
        let file = File::open(filename)
            .context(format!("Failed to open instance file `{}`", filename.display()))?;

        let mut reader = BufReader::new(file);
        let mut line = String::with_capacity(200);

        let mut line_counter = 0;

        let mut header = FxHashMap::default();

        while let Ok(bytes_read) = reader.read_line(&mut line) {
            line_counter += 1;

            if bytes_read == 0 || line.starts_with("NODE_COORD_SECTION") {
                break;
            }

            let mut split = line.trim().split(':');

            if let Some(key) = split.next() {
                if let Some(value) = split.next() {
                    header.insert(key.trim().to_owned(), value.trim().to_owned());
                }
            }

            line.clear();
        }

        let dimension = Self::get_header_value(&header, "DIMENSION")?;
        let num_of_items = Self::get_header_value(&header, "NUMBER OF ITEMS")?;

        let mut nodes = Vec::with_capacity(dimension);

        line.clear();
        let mut i = 1usize;

        while let Ok(bytes_read) = reader.read_line(&mut line) {
            line_counter += 1;

            if i > dimension || bytes_read == 0 {
                break;
            }

            let mut split = line.split_whitespace();

            let index = Self::get_section_value::<usize>(&mut split, "index", "node", line_counter)?;
            let x = Self::get_section_value::<f64>(&mut split, "x", "node", line_counter)?;
            let y = Self::get_section_value::<f64>(&mut split, "y", "node", line_counter)?;

            ensure!(index == i, "Index {index} in file is different from the index {i} expected while reading nodes.");
            i += 1;

            nodes.push(Node {
                index: index - 1,
                x,
                y,
                items: vec![],
                neighborhood: vec![],
                adjacency: vec![],
            });

            line.clear();
        }

        line.clear();
        let mut i = 1usize;
        let mut smallest_weight = f64::MAX;

        while let Ok(bytes_read) = reader.read_line(&mut line) {
            line_counter += 1;

            if i > num_of_items || bytes_read == 0 {
                break;
            }

            let mut split = line.split_whitespace();

            let label = Self::get_section_value::<usize>(&mut split, "index", "items", line_counter)?;
            let profit = Self::get_section_value::<f64>(&mut split, "profit", "items", line_counter)?;
            let weight = Self::get_section_value::<f64>(&mut split, "weight", "items", line_counter)?;
            let node = Self::get_section_value::<usize>(&mut split, "node", "items", line_counter)?;

            ensure!(label == i, "Index {label} in file is different from the index {i} expected while reading items.");

            i += 1;

            if weight < smallest_weight {
                smallest_weight = weight;
            }

            // Change it so the indexes starts counting from 0
            let node = node - 1;
            nodes[node].items.push(Item {
                index: label,
                profit,
                weight,
            });

            line.clear();
        }

        let mut items = Vec::with_capacity(num_of_items);

        for node in &mut nodes {
            for item in &mut node.items {
                items.push(item.index); // Here the indexes are the labels
                item.index = items.len() - 1; // Decrement so they become the indexes
            }
            node.items.shrink_to_fit();
        }

        let lin_kernighan = LinKernighan::new(&nodes, vendor_dir)?;

        let triangulation = lin_kernighan.triangulate(nodes.len())?;
        for (node, neighborhood) in triangulation.iter().enumerate() {
            let node = &mut nodes[node];

            for &neighbor in neighborhood {
                if neighbor != 0 && !node.neighborhood.contains(&neighbor) {
                    node.neighborhood.push(neighbor);
                }

                if neighbor != 0 && !node.adjacency.contains(&neighbor) {
                    node.adjacency.push(neighbor);
                }

                for &second_neighbor in &triangulation[neighbor] {
                    if second_neighbor != 0 && second_neighbor != node.index && !node.adjacency.contains(&second_neighbor) {
                        node.adjacency.push(second_neighbor);
                    }
                }
            }

            node.neighborhood.shrink_to_fit();
            node.adjacency.shrink_to_fit();

            node.neighborhood.sort_unstable();
            node.adjacency.sort_unstable();
        }

        let max_items_per_city = nodes.iter()
            .map(|n| n.items.len())
            .max()
            .unwrap_or_default();

        Ok(Self {
            nodes,
            items,
            smallest_weight,
            max_items_per_city,
            capacity_of_knapsack: Self::get_header_value(&header, "CAPACITY OF KNAPSACK")?,
            min_speed: Self::get_header_value(&header, "MIN SPEED")?,
            max_speed: Self::get_header_value(&header, "MAX SPEED")?,
            renting_ratio: Self::get_header_value(&header, "RENTING RATIO")?,
            lin_kernighan,
        })
    }

    pub(crate) fn speed_coefficient(&self) -> f64 {
        (self.max_speed - self.min_speed) / self.capacity_of_knapsack
    }
}
