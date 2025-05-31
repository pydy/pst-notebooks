# Mechanics Examples by Peter Stahlecker.

View the example gallery at:

https://pydy.github.io/pst-notebooks

## Set up the development environment

First, clone the repository and navigate to the directory in the terminal.

If you do not have the `pst-notebooks` conda environment, create it with:

```
conda env create -f environment.yml
```

If you do have the `pst-notebooks` conda environment, you may need to update it
with:

```
conda env update -f environment.yml
```

After the conda environment is created or updated, activate it with:

```
conda activate pst-notebooks
```

## Build the website

The primary Sphinx files in the directory are:

```
index.rst  # front page of the website, corresponds to index.html
Makefile  # commands to build the website on Linux/Mac
make.bat  # commands to build the website on Windows
conf.py  # Sphinx configuration file
gallery/plot_*.py  # example files, must start with "plot_"
gallery/GALLERY_HEADER.rst  # corresponds to example gallery page
```

Run:

```
make html
```

and then view the website locally by opening `_build/html/index.html` in your
web browser.

## Steps for updating the online website

1. Make changes to the source files in the `main` branch (i.e. `.rst` files or
   in `gallery/plot_*.py` files).
2. Run `make clean` to remove old website build output files.
3. Run `make html` to generate the website locally.
4. Check the `_build/html/index.html` locally in your web browser for errors.
5. If the website looks correct, commit all changes in the main branch.
6. Now upload a new version of the website to Github with the following
   command in the Conda Prompt:
   ```
   ghp-import --no-jekyll --no-history -m "Website update" -p _build\html`
