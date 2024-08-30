# Differentiable Model Predictive Excitation (DMPE):

This repository implements an algorithm for the excitation of systems with unknown (usually non-linear) systems.
The inner workings and lines of thought are outlined within the corresponding publication.
If you found this repository useful for your research, please cite it:

```
@Article{Vater**TBD**,
  author  = {**TBD**},
  journal = {**TBD**},
  title   = {**TBD**},
  year    = {**TBD**},
}
```
Excerpt from the abstract:

"The algorithm uses a model of the system in a model
predictive control fashion to predict the impact of inputs onto
the quality of the gathered data set. The inputs are optimized so
that system constraints are respected and a data quality metric
is minimized. This metric is designed to steer the data trajectory
to so far underrepresented facets of the systems state-action
space. The model is initialized in a random state and learned
in parallel to this excitation process from the gathered data." [Vater2024]


## Structure:

The repository consists of two main folders. `eval/` contains the code used in the experiments in [Vater2024]. `dmpe/` contains the source code for the DMPE algorithm and also for the GOATS algorithms from the related work.


## Basic Usage:

To apply the algorithms onto a system, the systems structure must comply to a specific API (Naturally, this can be adapted in the future to enable easier entry. Please open an issue, if you are interested in discussing this). Example environments following this API can be found in the `exciting_environments` repository.

...