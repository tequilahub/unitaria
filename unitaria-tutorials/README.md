# Tutorials Repository 

The purpose of this repo is to host tutorials of various functionalities of the tequila library. 

--> [see the rendered website here](https://tequilahub.github.io/tequila-tutorials/) <--

## Set Up
At first make sure to have an environment variable that you run the blog on. 

### Environment Variable 
To create one use:
```
conda create -n BlogQA python=3.9
```
After creating you can ensure this variable was created correctly and is among your existing ones through listing them with 
```
conda env list
```
To get into the variable and activate it run the following command followed by two other commands to install an ipkernal 
```
conda install ipykernel
conda activate BlogQA
python -m ipykernel install --user --name=BlogQA
```

### Running 
For running the site with quarto run 
```
quarto preview 
```
Now some packages may not be installed yet. For this, install all packages required each time *quarto preview* shows a missing one. 
Here is a list of some important and required ones:
```
pip install tequila-basic
pip install pyyaml
pip install nbformat
pip install nbclient
pip install numpy
pip install scipy
pip install matplotlib
pip install qulacs
conda install psi4
pip install pyscf
```
Also, these packages are required for the Sphinx documentation:
```
pip install sphinx
pip install sphinxcontrib
pip install sphinxcontrib-programoutput
pip install sphinx-rtd-theme 
```

# Quarto and LaTeX and the Markdown language - for Tutorials

sometimes quarto misbehaves when trying to format text in one of these to manners. 
If you are working with a Jupyter Notebook in your tutorial and wish to embed LaTeX or markdown in it, make sure that:
- you use markdown cells for the markdown parts 
- LaTeX should be possible everywhere, with some exceptions and under usage of some *weird* rules described below, so pay attention:

1. inline LaTeX usage should be used like this: $ content$ 
    - notice the space after the first dollar and the missing space before the second one
    moreover, if it still doesn't work, try another two spaces after the closing dollar 
    - another way of seeing the spaces is: $"space"content$"space""space" 
2. when using enumerations of any kind, i.e. numerical or with bullet points you should make two spaces and in the end of each line and and hit enter.
In order for the spaces and for enter to be visible and understandable they are described like so: "space", "enter"
- example: 
1."space" my enumeration"space""space""enter"
