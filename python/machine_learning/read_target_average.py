"""Paper Table 2: number of data points and mean formation energy / zero-phonon line per dataset."""

from common import DEFECT_TYPES, DESCRIPTORS, load_datasets

print(f"{'descriptor':28} {'defect type':24} {'single: N':>9} {'FE':>6} {'ZPL':>7} | {'double: N':>9} {'FE':>6} {'ZPL':>7}")
for descriptor in DESCRIPTORS:
    for defect_type in DEFECT_TYPES:
        data, data_single = load_datasets(descriptor, defect_type, columns=["FE", "ZPL"])
        print(f"{descriptor:28} {defect_type:24} {len(data_single):9d} {data_single.FE.mean():6.2f} {data_single.ZPL.mean():7.4f}"
              f" | {len(data):9d} {data.FE.mean():6.2f} {data.ZPL.mean():7.4f}")
