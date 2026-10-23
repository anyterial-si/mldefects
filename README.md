# Pre-Screening Point Defects in Semiconductors With Machine Learning

Applying machine learning to point defects in semiconductors to predict formation energies and zero-phonon lines.

File layout:
```
data
machine_learning
└── saved_models
results
└── sic_no_wyckoff
    └── interstitial
        └── FE
        └── ZPL
    └── vacancy-subsitutional
        └── FE
        └── ZPL
└── sic_wyckoff
    └── interstitial
        └── FE
        └── ZPL
    └── vacancy-subsitutional
        └── FE
        └── ZPL
└── sic_wyckoff_letter
    └── interstitial
        └── FE
        └── ZPL
    └── vacancy-subsitutional
        └── FE
        └── ZPL
└── sic_wyckoff_letter_nw_data
    └── interstitial
        └── FE
        └── ZPL
    └── vacancy-subsitutional
        └── FE
        └── ZPL
└── sic_wyckoff_nw_data
    └── interstitial
        └── FE
        └── ZPL
    └── vacancy-subsitutional
        └── FE
        └── ZPL
```

The top level directories are described in separate sections below.

## Data

Contains the created CSV files with data extracted from the ADAQ database used for this work, as well as the descriptor vector duplicates

### Machine learning scripts

Scripts for training and inference from the models.

- `correlation_heatmaps.py`: used to create the plots of the covariance and the pearson correlation coefficient of the input parameters

- `data_average.py`: computes the average of the number of data points used in each learning curve point, need to be run before plot.py (already have)

- `init_ml.py`: used to initialize the ml.py script with imports and argparse

- `input_parameters_details.py`: used to plot the standard deviation, mean, max, min of all the input parameters in the different descriptor datasets

- `ml.py`: creates the data for the learning curves and the scatter plots

- `neural_heatmaps.py`: used to plot the heatmaps showing the optimization of the neural model

- `optim_kernel.py`: used for the optimization of the kernel ridge model, buggy when saving optim alpha and kernel, plot.py fixes this

- `optim_linear.py`: optimize linear

- `optim_neural.py`: optimize neural

- `plot.py`: creates the plots of the learning curves, the scatter plots, the heatmaps showing the optimization of the kernel ridge model, and the plots showing the optimization of the linear model

- `predict.py`: predicts the energies in the ../database/extend_data_cluster .csv files, uses saved models and scalings in folder ./saved_models

- `read_target_average.py`: used to check the average formation energy and zero-phonon line of the different descriptor datasets and the number of data points in each

- `schedule.py`: script to run data_average.py and ml.py in a series for the different cases

### results

Contains the data for the point defects for the **different descriptors** each sorted in its own subdirectory:

- `sic_no_wyckoff` - results for the "Atomic Descriptor" in the paper

- `sic_wyckoff_letter` - results for Wyckoff Letter Descriptor in the paper

- `sic_wyckoff` - results for Wyckoff Coordinate Descriptor in the paper

- `sic_wyckoff_letter_nw_data` - results for Wyckoff Letter Compare Descriptor in the paper

- `sic_wyckoff_nw_data` - results for Wyckoff Coordinate Compare Descriptor in the paper

Each descriptor contains a subdirectory for **respective type of defect** with the CSV files for the respective point defect data for the corresponding descriptor:

- `vacancy-substitutional` - results for vacancies and substitutions.
- `interstitials` - results for defect cluster containing at least one interstitial.

Each defect type directory contains a directory per **target variable**:

- FE - formation energy
- ZPL - zero-phonon line

They contain:

* Optimization results for the ridge and the kernel ridge model for different scalings ('' meaning nothing means standardized). Contains the data for the plotting of the optimizations:

    ```
    optim_results_linear_normal.toml
    optim_results_linear.toml
    optim_results_linear_unscaled.toml

    optim_results_kernel_normal.toml
    optim_results_kernel.toml
    optim_results_kernel_unscaled.toml
    ```

* Databases for the multilayer perceptron model optimization. Used for the plotting of the optimization. .clean is for double-checking the .db results:

    ```
    optim_results_neural_clean.db
    optim_results_neural.db
    optim_results_neural_normal.db
    ```

* Scatter data for the different models used to plot the scatter plot:

    ```
    results_scatter.toml
    ```

* For more detailed optimization of the kernel model, not used when the gains were realized to be abyssmal, ignore:

   ```
   results_10_fold.toml
   results.toml
   optim_results_kernel_finer.toml
   optim_results_kernel_unscaled_finer.toml
   ```

**kernel_partial_results and neural_partial_results folder**

contains .toml files of 'parallelized' ml.py runs on the supercomputer for the kernel or the neural model. Used to plot the learning curves
