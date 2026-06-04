# LISA SBI tutorial

This is a toy-model for a SBI pipeline in LISA. 
Everything is much simplified and only intended to understand the basic mechanisms that need to be set up in `jax`.

## Setting-up the environment

First `cd` into the relevant directory where you want to install your `venv`. 
I would recommend installing `uv` via `brew install uv` (on mac). 


```
uv venv
source .venv/bin/activate
```

Then you need to install some packages like

```
uv pip install jax flowjax equinox ipykernel matplotlib corner
```

Finally, intall the package via 
```
uv pip install -e .
```






## Fill-in notebook

If you want to learn about the code interactively, try having a look at `01a_simplest_sbi_example_FILL.ipynb` first. 
Once you have tried, the completed notebook can be found in `01b_simplest_sbi_example.ipynb`. 

## Notebooks for exploring

If you want to see the training directly, see `02a_training.ipynb`. 
The training should take 10 seconds on a laptop. 
If you want to play directly with a trained model, see `02b_inference_only.ipynb`. 