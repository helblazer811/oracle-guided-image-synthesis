# [Oracle Guided Image Synthesis with Relative Queries](https://openreview.net/forum?id=rNh4AhVdPW5)

This is a repository for my work on the paper ["Oracle Guided Image Synthesis with Relative Queries"](https://openreview.net/forum?id=rNh4AhVdPW5).

<img src="readme_images/explanatory_figure.png">
<img src="readme_images/full_video.gif">

Link: https://openreview.net/forum?id=rNh4AhVdPW5

If you found this paper interesting please cite using the following bibtex:

```bibtex
@inproceedings{
  helbling2022oracle,
  title={Oracle Guided Image Synthesis with Relative Queries},
  author={Alec Helbling and Christopher John Rozell and Matthew O'Shaughnessy and Kion Fallah},
  booktitle={ICLR Workshop on Deep Generative Models for Highly Structured Data},
  year={2022},
  url={https://openreview.net/forum?id=rNh4AhVdPW5}
}
```

## Code setup

Create a conda environment from the requirements.txt file. 

```
    conda create --name <env> --file requirements.txt
```

## Run a basic experiment

You can run one of our experiment templates as follows. Each contain a python dictionary, which configures the model, dataset, and experiment. 

1. ```cd auto_localization/experiments/morpho_mnist```
2. ```python bayesian_triplet_experiment.py <run_name>```

## Experiment Analysis

You can analyze these models using the jupyter notebooks in ```auto_localization/experiments/morpho_mnist/experiment_analysis/```.
