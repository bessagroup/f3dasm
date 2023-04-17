# Sphinx documentation for f3dasm

This folder contains the Sphinx documentation for the `f3dasm` package. The documentation is generated using Sphinx, a tool that makes it easy to create beautiful documentation for Python projects.

## Building the Documentation Locally

To build the documentation locally, you'll need to have the full version of `f3dasm`, Sphinx and it's extensions installed on your system. You can install these by installing the development requirements `requirements_dev.txt`:

```bash
pip install -r requirements_dev.txt
```

Once you have that installed, navigate to this folder in your terminal and run the following command:

```console
make html
```


This will generate the HTML version of the documentation in a new `build` directory. You can open the documentation in your web browser by opening the `build/html/index.html` file.

Creating the documentation and automatically opening the web-page can be done with the following command:

```console
make html-open
```

## Contributing to the Documentation

If you notice any issues with the documentation or would like to contribute to it, please feel free to open a pull request on the [f3dasm GitHub repository](https://github.com/bessagroup/f3dasm) or contact the maintainers directly.
