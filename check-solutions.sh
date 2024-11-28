for file in ./results/*.flns; do
  echo checking file "$file"

  instance=(${file//./ })
  instance=(${instance//// })
  instance=${instance[-1]}

  if ! ./target/release/flns -xc "$file" instances/"$instance".ttp; then
    exit 1
  fi

  echo
done

echo "All valid."
