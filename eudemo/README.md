# EU-DEMO Reactor Design

__Work in progress: this model is not yet fully functional__

This folder contains the EU-DEMO reactor design.
To use the `eudemo` package, you must add its path to your Python path:

```bash
export PYTHONPATH="<path/to/bluemira>/eudemo:${PYTHONPATH}"
```

To run the reactor build,
`cd` into the `eudemo` directory and run the `reactor.py` file:

```bash
cd <path/to/bluemira>/eudemo
python eudemo/reactor.py
```

The `cd` is required, as the paths in the build config are
relative to the `eudemo` directory.

In future this will be moved to a separate repository.
It should be used as a template for how we expect
other reactor repositories to be structured.
