# EDBOplus rebuild

Optimizing reactions using Multi-Objective Bayesian optimization. Based on the paper [A Multi-Objective Active Learning Platform and Web App for Reaction Optimization](https://pubs.acs.org/doi/10.1021/jacs.2c08592?ref=pdf) and the source [code](https://github.com/doyle-lab-ucla/edboplus) 
## Setup
Clone the project :

    git clone https://github.com/DeepSynthesis/rebuild_EDBOplus.git


Create and activate the environment:

    conda create -n py38 python=3.8
    conda activate py38

Install the dependencies:

	pip install -r requirements.txt

## Usage
refer to [summit-domain](https://gosummit.readthedocs.io/en/latest/domains.html#)

[summit-CategoricalVariable](https://gosummit.readthedocs.io/en/latest/_modules/summit/domain.html#CategoricalVariable)

Add a discrete variable:
- name：variable name
- description：the description of the variable
- level： a level to the discrete variable
- descriptors：class:`~summit.utils.dataset.DataSet`, optional A DataSet where the keys correspond to the levels and the data columns are descriptors.


```python
domain += CategoricalVariable(
                name=x, description=x, levels=lev_v, descriptors=data_v
            )
```
----

[summit-ContinuousVariable](https://gosummit.readthedocs.io/en/latest/_modules/summit/domain.html#ContinuousVariable)

Add a objective that needs to be optimized:
- bounds : the range of the objective
- maximize : maximize(True) or minimize(False) the objective

```python
domain += ContinuousVariable(
        name=obj, description=obj, bounds=[-100, 100],
        is_objective=True, maximize=True,
    )
```
----

Build a domain of the reaction conditions:

```python
strategy = newEDBO(domain, sobol_num_samples=512, init_method='LHS')
```

Suggest new conditions based on previous data points：

```python
next_points = strategy.suggest_experiments(prev_res=prev_res, batch_size=batch_size)
```
## Example
`1728_test.py`：

[HTE data set](https://github.com/doyle-lab-ucla/edboplus/tree/main/examples/publication/BMS_yield_cost/data) consists of 1728 total conditions

	python 1728_test.py --batch_size=5 --seed=1234 --init_method=LHS

- init_method：LHS or CVT
- batch_size：1, 2, 3 or 5

----

`352_test.py` ：

[HTE data set](https://github.com/doyle-lab-ucla/edboplus/tree/main/examples/publication/Suzuki/data) consists of 352 total conditions

	python 352_test.py --batch_size=5 --seed=1234 --init_method=LHS --encode=DFT

- init_method: LHS or CVT
- batch_size：1, 2, 3 or 5
- encode: DFT, mordred or OHE 

----

`106560_test.py` :

The condition space size is 106560 and there are already 39 data points.

Run the following commands to give five new conditions:

 	python 106560_test.py
 	
