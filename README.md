# Model-Agnostic Meta-Learning (MAML)
Here we implement the MAML algorithm from the paper ["Model-Agnostic Meta-Learning for Fast Adaption of Deep Networks"
by Chelsea Finn, Pieter Abbeel and Sergey Levine](https://arxiv.org/abs/1703.03400) within the Spinning Up framework.

We aim at reproducing their results in the 2D Navigation environment (Figure 4 in the paper). 

The contents of this repository were created in the scope of the project "Deep Reinforcement Learning" at Ulm University.

## Install
Follow the Spinning Up installation guide on https://spinningup.openai.com/en/latest/user/installation.html.
It is not required to install MuJoCo.
- set up a python 3.6 environment (in our case it was python 3.6.9)
- install OpenMPI: `sudo apt-get update && sudo apt-get install libopenmpi-dev`
- install Spinning Up (the state of the github repository when cloning it was: 038665d on 7 Feb 2020):  
`git clone https://github.com/openai/spinningup.git`  
`pip install -e spinningup` (in case you get an error execute `python -m pip install --upgrade pip` and try again, pip version was then 21.3.1)


Copy the `maml` folder into the `spinningup/spinup` directory and replace the `spinningup/spinup/run.py` and `spinningup/spinup/utils/plot.py` files
with the respectively named `.py` files.

Install the gym-twoDNavigation environment: `pip install -e gym-twoDNavigation`

If you want to search for learning rates, hyperopt is required: `pip install hyperopt==0.2.7`

To be on the save side, we here list the version of all installed libraries that we used (they are not set for every library in the Spinning Up setup script)
- cloudpickle==1.2.1
- gym[atari,box2d,classic_control]==0.15.7
- ipython==7.16.3
- joblib==1.1.0
- matplotlib==3.1.1
- mpi4py==3.1.3
- numpy==1.18.5
- pandas==1.1.5
- pytest==6.2.5
- psutil==5.9.0
- scipy==1.5.4
- seaborn==0.8.1
- tensorflow==1.15.5
- torch==1.3.1
- tqdm==4.62.3

## Running the Experiments
Switch to the directory `spinningup/spinup/maml/pytorch/` and use the `run_experiment.py` file to run the MAML experiments.  
There is a commandline interface provided. In order to see how to use it, run: `python run_experiment.py -h`.

The output directory is `spinningup/data/<train or evaluate>-<algorithm name>`.

If you want to run the same experiments as in our replication study, you can have a look at the `ExperimentScript.pdf` file,
which was used by us in order to keep track of the experiments to be run.

### Visualisation provided by Spinning Up
You can plot the results via Spinning Up as described here https://spinningup.openai.com/en/latest/user/plotting.html.
Additionally, we have added the following parameters:
- `--yscale`: define the scale for the y-axis, e.g. `symlog`.
- `--ylim`: define the limits of the y-axis, e.g. `-100 -3`.
- `--ci`: switches from standard deviation to the specified confidence interval, e.g. `95` for plotting the 95% confidence interval.

```
python -m spinup.run plot [path/to/output_directory ...] [--legend [LEGEND ...]]
    [--xaxis XAXIS] [--value [VALUE ...]] [--count] [--smooth S]
    [--select [SEL ...]] [--exclude [EXC ...]]
	[--yscale SCALE] [--ylim LOWER UPPER] [--ci CI] 
```

You can also run the trained policy via Spinning Up as described here
https://spinningup.openai.com/en/latest/user/saving_and_loading.html#loading-and-running-trained-policies.
```
python -m spinup.run test_policy path/to/output_directory
```

## Sources
This implementation is build upon the Spinning Up Framwork by OpenAI (http://openai.com) which is licensed under
the MIT License (for more details see maml/LICENSE file).

Each file contains a "header", in which we state:
- on which file from Spinning Up the implementation is based on
- what we changed in the file
- what are the sources for this change if any

Please, see also the `gym-twoDNavigation/README.md` file for more information on the sources used for the implementation of the environment.
