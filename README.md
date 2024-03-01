## icenet-notebooks

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)

This repository contains instructional notebooks that describe how to use the library and pipelining utilities built around it. 

### Instructional notebooks

_Currently only notebooks 01 and 02 are completely tested_

The notebooks 01-05 are intended for instruction and contain output from example runs.

Notebooks xx.\* are illustrative and exemplify/contain runs from the operational infrastructure, but rely on the model run(s) they were run against. These are easily adaptable for your own local executions with IceNet.

#### Status / Compatibility

__These are compatible at time of writing with the 0.2.\* version of the [IceNet library](https://github.com/icenet-ai/icenet).__

### scratch\_ noteboooks

Many of the other notebooks are scratch or potentially useful (or were once useful) when the codebase was being refactored. These are named scratch\_ to show that they're only potentially useful if you want to explore or adapt them.

### Other notebooks

Any other notebook is likely related to events or presentations that have been given.

## Contributing

Please fork and raise PRs, _contributions are most welcome!_

### Removing notebook outputs

Sometimes it's a good idea to strip the output from notebooks prior to commiting changes and opening pull requests, to make it easier to compare.

This can be done by manually removing the output in the Jupyter notebooks, but is simpler using the `nbstripout` tool running as a pre-commit hook.

This requires installation of `nbstripout` and `pre-commit` via `pip install nbstripout pre-commit` running in your conda environment (note that both can alternatively be installed from conda-forge).

Finally, install the pre-commit hook by running `pre-commit install` (making sure to be in your clone of this repository) which will install the pre-commit hooks defined in the `.pre-commit-config.yaml` file in this repository.

Now making any commits to `*.ipynb` files (i.e. jupyter notebooks) will run `nbstripout` against them before commiting.

