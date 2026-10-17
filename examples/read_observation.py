"""Read an RDEF observation file and the product files it lists."""

from pathlib import Path

from rdefpy import read_observation, read_product

data = Path(__file__).parents[1] / "tests" / "data"
obs = read_observation(data / "EXMPn000tIsDS24r01c00-26039120000.obs")
print(obs)

for scan in obs:
    print()
    print(scan)
    for entry in scan.products:
        print()
        print(read_product(entry.datafile))
