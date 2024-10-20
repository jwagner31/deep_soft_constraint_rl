# Deep Max Entropy Soft Constraint Inverse Reinforcement Learning

You can find some examples from the grid-world in the notebooks folder.

## Installing the requirements

```bash
  pip install -r requirements.txt
```

## Running the experiments

### Hard constraints

To learn the constraints run:

```bash
  python -m max_ent.examples.learn_hard_constraints
```

After learning, run the following to generate the reports (in [`./reports/hard/`](reports/hard) folder):

```bash
  python -m max_ent.examples.compare_hard_results
```

### Soft constraints

To learn the constraints run:

```bash
  python -m max_ent.examples.learn_soft_constraints
```

After learning, run the following to generate the reports (in [`./reports/soft/`](reports/soft) folder):

```bash
  python -m max_ent.examples.compare_soft_results
```

### Transfer Learning
To run the transfer learning experiments and generate the results use:

```bash
  python -m max_ent.examples.transfer
```
The generated reports can be found in [`./reports/transfer/`](reports/transfer) folder.

### Orchestration

Run the notebook in `./notebooks/new_metrics.ipynb` .

Also, you can set `learn = True` in `./max_ent/examples/orchestrator_exp.py` then run:

```bash
  python -m max_ent.examples.orchestrator_exp
```

After that, set `learn = False` and run the above command again.
The reports will be generated into `./reports/orchestrator/` folder.

## Acknowledgement

This repository is an extension of MESC-IRL (Max Entropy Soft Constraint Inverse Reinforcement Learning) algorithm. The original code can be found [here](https://github.com/Rahgooy/soft_constraint_irl).

This repository uses and modifies some codes from [irl-maxent](https://github.com/qzed/irl-maxent) library.
