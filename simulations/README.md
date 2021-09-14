# Simulations

## Set up

This code requires Python 3, `cvxpy`, `gurobi`, `pypoman`, `numpy`,`matplotlib`, `scipy`.


## Running
```
python3 main.py
```

## Structure
The code for constructing the FTOCP is in `ftocp.py` and the code for the low-level controllers is in `unicycle.py` and `low_level_controllers.py`. The specific parameters, like $$P$$ is determined using `ricatti_sols.py`