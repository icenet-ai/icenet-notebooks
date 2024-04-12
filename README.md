## icenet-notebooks

This repository contains instructional notebooks that describe how to use the [IceNet library](https://github.com/icenet-ai/icenet) and [pipelining utilities](https://github.com/icenet-ai/icenet-pipeline) built around it.

### Instructional notebooks

_Currently, notebooks 01, 02, 03, 04 are completely tested_

The notebooks 01-05 are intended for instruction and contain output from example runs.

Notebooks xx.\* are illustrative and exemplify/contain runs from the operational infrastructure, but rely on the model run(s) they were run against. These are easily adaptable for your own local executions with IceNet.

A brief description of the different notebooks included in this repository:

* [01.cli_demonstration.ipynb](01.cli_demonstration.ipynb): Shows the usage of command line interfaces (with [IceNet library](https://github.com/icenet-ai/icenet) installed) for end-to-end runs to forecast sea-ice concentration.

* [02.pipeline_demonstration.ipynb](02.pipeline_demonstration.ipynb): Shows the usage of the [IceNet pipeline](https://github.com/icenet-ai/icenet-pipeline)) for end-to-end runs and ensemble modelling, as deployed to a SLURM cluster (A cluster is not mandatory, it can also be run locally).

* [03.data_and_forecasts.ipynb](03.data_and_forecasts.ipynb): Shows the the various data sources, intermediaries and products that arise from both of the first two notebooks [CLI demonstrator notebook](01.cli_demonstration.ipynb) and [Pipeline demonstrator notebook](02.pipeline_demonstration.ipynb) activities.

* [04.data_and_forecasts.ipynb](04.library_usage.ipynb): Shows the usage of the [IceNet library](https://github.com/icenet-ai/icenet) via its Python API.

#### Status / Compatibility

__These are compatible at time of writing with the 0.2.\* version of the [IceNet library](https://github.com/icenet-ai/icenet).__

Please ensure you have installed `netCDF4<=1.6` in your Python environment. You might run into issues on netCDF outputs with newer versions.

```bash
pip install -U netCDF4<=1.6.0
```

If you are running into setup issues, please use the conda `environment.yml` file from [icenet-pipeline](https://github.com/icenet-ai/icenet-pipeline) corresponding to the tagged version of this repository to create a python conda environment.

All three main icenet-related repositories ([icenet](https://github.com/icenet-ai/icenet), [icenet-notebooks](https://github.com/icenet-ai/icenet-notebooks) and [icenet-pipeline](https://github.com/icenet-ai/icenet-pipeline)) share a common version tag.

### scratch\_ noteboooks

Many of the other notebooks are scratch or potentially useful (or were once useful) when the codebase was being refactored. These are named scratch\_ to show that they're only potentially useful if you want to explore or adapt them.

### Other notebooks

Any other notebook is likely related to events or presentations that have been given.

## Contributing

Please fork and raise PRs, _contributions are most welcome!_
