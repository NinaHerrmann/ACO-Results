# Results of exemplary ACO Implementation
This repository collects the results from performance measurements of a program implementing a version of the ACO method. One program is hand written in CUDA refered to as low-level implementation. One program is written in the DSL Musket. The repository is structured as follows:
1. the folder raw_data includes complete runtimes with multiple runs for multiple settings,
2. the folder data_aggregation has files with the average of runtimes formultiple runs with the same setting,
3. the folder archive_results includes old runtimes with multiple runs for multiple settings,
4. the folder plot_scripts include scripts to generated graphical representation of the data,
5. the folder WIP_accumulate_data_scripts contains python scripts to calculate the average of multiple runs (however, since the procedure depends on the way the data is generated they often require adjustments). 
