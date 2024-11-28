if ! rustup toolchain list; then
  echo "Installing Rust..."
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
  source "$HOME/.cargo/env"
fi

if ! rustup toolchain list | grep -q 'nightly-x86_64-unknown-linux-gnu'; then
  echo "Installing Rust nightly..."
  rustup toolchain install nightly
fi

if ! gcc --version; then
  echo "Installing GCC"
  sudo apt-get update
  sudo apt-get install -y build-essential
fi

echo "Downloading Concorde..."
rm -r vendor
mkdir vendor
cd vendor || exit 1
wget https://www.math.uwaterloo.ca/tsp/concorde/downloads/codes/src/co031219.tgz
tar -xzf co031219.tgz

echo "Compiling Concorde..."
cd concorde || exit 1
./configure
make

cp LINKERN/linkern ..
cp EDGEGEN/edgegen ..

cd ../../
chmod +x ./vendor/linkern
chmod +x ./vendor/edgegen

echo "Building program..."
cargo build --release

echo "Build successfully"
echo "You should have a 'flns' executable in './target/release/"
echo "To run execute './target/release/flns <PROBLEM_FILE>'"
