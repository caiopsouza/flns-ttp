# Fast Large Neighborhood Search for the Travelling Thief Problem

This project implements a heuristic to solve the [Travelling Thief Problem](https://ieeexplore.ieee.org/abstract/document/6557681).

# Quick Start

This software was developed in Linux Mint 21.1 Cinnamon and should work in any Linux distribution.

Run the `installation.sh` script to install and build everything or alternatively:

- [Install Rust nightly](https://www.rust-lang.org/tools/install).
- Download the `Concorde-03.12.19`version of the [Concorde TSP Solver](https://www.math.uwaterloo.ca/tsp/concorde/downloads/downloads.htm).
- Compile it [following these instructions](https://www.math.uwaterloo.ca/tsp/concorde/DOC/README.html).
- Copy the files `LINKERN/linkern` and `EDGEGEN/edgegen` to the `vendor` folder inside this repo. Make sure they're executable.
- Run `cargo build --release` to create the executable.

To solve an instance, run `./target/release/flns <PROBLEM_FILE>` with your instance file as parameter. It'll run for 10 minutes and continually output the best solution found so far in the current directory. 

Detailed description on how to build and run below.

# Requirements

A script to install the requirements is available in `installation.sh`, you might need to edit it if you use a different package manager from `apt`.

The script will install Rust, GCC, download the dependencies, compile everything and create an executable called `flns` in `/target/release`.

If you want to configure the environment on your own, follow these steps.

## Rust

You'll need the nightly version of Rust to compile.

Run: `$curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh` as instructed in the [Rust install page](https://www.rust-lang.org/tools/install).
 
- When prompt, type `2` to choose the option `2) Customize installation`.
- `Default host triple?`. Press enter to choose the default option.
- `Default toolchain?`. Type `nightly` and press enter.
- `Profile (which tools and data to install)?`. Press enter to choose the default option.
- `Modify PATH variable?`. Press enter to choose the default option.
- The installer will confirm if everything is correct, press enter if so or type `2` and repeat these steps.

## Concorde

You'll also need the `concorde` software to solve the TSP portion of the problem and find the edge set.

- Download and extract the `Concorde-03.12.19` source from the [download page](https://www.math.uwaterloo.ca/tsp/concorde/downloads/downloads.htm).
- Install `gcc` if not already present: `#apt install build-essential`.
- Following the [instructions on the manual](https://www.math.uwaterloo.ca/tsp/concorde/DOC/README.html), simply run `./configure`, then `make`.
- Copy the files `LINKERN/linkern` and `EDGEGEN/edgegen` to the `vendor` folder inside this repo.

# Build

After installing all requirements, simply run `cargo build --release` to create the `flns` executable in `./target/release/`.

# Run

To run type: `./target/release/flns <PROBLEM_FILE>`.

The program will run for 10 minutes and create a file on the working directory with the best solution found so far.

You can supply the `-o <OUTPUT_DIR>` to specify a different output for the solutions or `-h` for additional options.

This program continually save the best solution found so far on the same file, Interrupting it might corrupt the solution if it's in the middle of a writing.