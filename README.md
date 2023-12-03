
# Biologically Informed Deep Neural Network to Interpret Non-coding Chromatin Interactions

This repository contains the source code and data files for a biologically informed deep neural network designed for the interpretation of enhancer-promoter interactions.

## Getting Started

To set up a local copy and run the project, follow these simple steps:

### Prerequisites

Make sure you have Python 3 installed. You can find the list of required packages in the `environments.yml` file.

### Installation

1. Clone the repository:

   ```
   git clone https://github.com/iced-soda/EPNET
   ```

2. Depending on your intended use, you might need to download one or more of the following:

   a. **Data Files:** These are required for retraining models and generating figures. Extract the data files under the `_database` directory. If you prefer to store them in a different location, adjust the `DATA_PATH` variable in `config_path.py` accordingly.

   b. **Log Files:** These are needed to regenerate figures from the paper. Extract the log files under the `_logs` directory. If you want to use a different directory, modify the `LOG_PATH` variable in `config_path.py`.

   c. **Plot Files:** This directory contains copies of the paper's images. Extract these files under the `_plots` directory. If you wish to store them elsewhere, make changes to the `PLOTS_PATH` variable in `config_path.py`.

## Usage

Run the `simple_training.ipynb` notebook.

## Contributing


## License


## Acknowledgments


---
