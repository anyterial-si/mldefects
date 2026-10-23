import subprocess

materials = [
    "sic_wyckoff",
    "sic_wyckoff_letter",
    "sic_no_wyckoff",
    "sic_wyckoff_letter_nw_data",
    "sic_wyckoff_nw_data",
]
defect_types = [
    "vacancy-substitutional",
    "interstitial"
]
target_variables = [
    "FE",
    "ZPL",
]
scalings = [
    "unscaled",
    "standard",
    "normal",
]
for material in materials:
    for defect_type in defect_types:
        for target_variable in target_variables:
            for scaling in scalings:

                print(f"##### {material} {defect_type} {target_variable} {scaling} #####")

                # print("\n optim linear.py standard \n")
                # result = subprocess.run(f"time python3 optim_linear.py --scaling standard --material '{material}' --defect_type '{defect_type}' --target_variable '{target_variable}'", shell=True, capture_output=True, text=True)
                # print(result.stdout)
                # print(result.stderr)
                # print("\n optim linear.py unscaled \n")
                # result = subprocess.run(f"time python3 optim_linear.py --scaling unscaled --material '{material}' --defect_type '{defect_type}' --target_variable '{target_variable}'", shell=True, capture_output=True, text=True)
                # print(result.stdout)
                # print(result.stderr)
                # print("\n optim linear.py normal \n")
                # result = subprocess.run(f"time python3 optim_linear.py --scaling normal --material '{material}' --defect_type '{defect_type}' --target_variable '{target_variable}'", shell=True, capture_output=True, text=True)
                # print(result.stdout)
                # print(result.stderr)

                # print('\n ml.py \n')
                # result = subprocess.run(f"time python3 ml.py --cross_validation 'varied' --scaling {scaling} --model linear --material '{material}' --defect_type '{defect_type}' --target_variable '{target_variable}'", shell=True, capture_output=True, text=True)
                # print(result.stdout)
                # print(result.stderr)

                # print('\n ml.py \n')
                # result = subprocess.run(f"time python3 ml.py --cross_validation '10-fold' --scaling normal --model linear --material '{material}' --defect_type '{defect_type}' --target_variable '{target_variable}'", shell=True, capture_output=True, text=True)
                # print(result.stdout)
                # print(result.stderr)


                # print('\n data_average.py \n')
                # result = subprocess.run(f"time python3 data_average.py --material '{material}' --defect_type '{defect_type}' --target_variable '{target_variable}'", shell=True, capture_output=True, text=True)
                # print(result.stdout)
                # print(result.stderr)

                # print('\n data_average.py varied \n')
                # result = subprocess.run(f"time python3 data_average.py --cross_validation '10-fold' --material '{material}' --defect_type '{defect_type}' --target_variable '{target_variable}'", shell=True, capture_output=True, text=True)
                # print(result.stdout)
                # print(result.stderr)

                print('\n correct_kernel \n')
                result = subprocess.run(f"time python3 correct_kernel_optim.py --material '{material}' --kernel_optim_target '_unscaled' --defect_type '{defect_type}' --target_variable '{target_variable}'", shell=True, capture_output=True, text=True)
                print(result.stdout)
                print(result.stderr)
