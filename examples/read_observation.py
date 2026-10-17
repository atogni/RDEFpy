"""Read an RDEF observation file and the product files it lists."""

from pathlib import Path

from rdefpy import read_observation

try:
    from rdefpy import read_product
except ImportError:  # product-file support not installed
    read_product = None

data = Path(__file__).parents[1] / "tests" / "data"
obs = read_observation(data / "EXMPn000tIsDS24r01c00-26039120000.obs")
print(obs)

for scan in obs:
    print()
    print(scan)
    if read_product is None:
        continue
    for entry in scan.products:
        print()
        print(read_product(entry.datafile))
