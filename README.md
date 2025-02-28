# Synergistic Cobalt/Enamine Catalysis Reaction Discovery and Optimization

## Project Overview
This project employs a heuristic data-driven strategy to discovery an optimal catalytic system for the enamine-Co(IV) catalysis. Using an ML-driven optimization loop, we screened over 100,000 conditions and found the optimal condition in just 64 experiments. Furthermore, a clustering-based analysis facilitated a systematic assessment of substrate generality, confirming the broad applicability of the catalytic mode. 

## Table of Contents
- [Installation](#installation)
- [Running](#running)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Installation
To set up the environment and install the necessary dependencies, follow these steps:

1. Create a new Conda environment with Python 3.10:

   ```bash
   conda create -n react_opt python=3.10 -y
   ```

2. Activate the new environment:

   ```bash
   conda activate react_opt
   ```

3. Install the dependencies from requirements.txt:

    ```bash
   pip install -r requirements.txt
   ```

4. Some bug fix of `summit` package：change `<CONDA_ENV_PATH>/lib/python3.10/site-packages/summit/benchmarks/experimental_emulator.py`. 
    - Comment out the `_check_fit_params` parameter imported on line 40
    - Comment out `from sklearn.utils.fixes import delayed` on line 43

## Running

see more in [`demo.ipynb`](./demo.ipynb)

## Contributing
We welcome contributions from the community. Please fork the repository and submit pull requests.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact
For any questions or inquiries, please contact us at [tzz24@mails.tsinghua.edu.cn](mailto:tzz24@mails.tsinghua.edu.cn).