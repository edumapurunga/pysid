# Development

First of all, thank you for your interest in improving this project.   
Please, follow these guidelines in order to minimize efforts for all people involved.  

For testing and development purposes and to contribute with code please follow the following instructions.

We sharply recommend to create a dedicated development environment for your python with the exact requisites for running pysid.   

1. If you are a first-time contributor:
   - Go to https://github.com/edumapurunga/pysid and click "fork" button to create your own copy of the project.
   
	- Clone the project to your local computer:
   
     ```bash
     git clone https://github.com/your-user-name/pysid.git
     ```
   
   - Change directory:
   
     ```bash
     cd pysid
     ```
   
   - Add the main repository:
   
     ```bash
     git remote add main-pysid https://github.com/edumapurunga/pysid.git
     ```
   
   - Now you should have the code on your local computer and your remote repositories should be your personal repository on github and the main pysid repository. Check this by using the following command:
   
     ```bash
     git remote -v 
     ```
   
2. Develop your code:

   - Pull the latest changes from pysid:

     ```bash
     git checkout master
     git pull main-pysid master
     ```

   - Create a branch for the feature you want to work on the code. Please use a related name for which one can understand which features are being merged in the code. For instance: method-instrumental-variable, doc-add-html. 

     ```bash
     git checkout -b method-instrumental-variable
     ```

   -  Remember to locally commit  all of your changes (git add and git commit). Use a properly formatted commit message, write tests that fail before your changes and pass afterward. Run all tests locally and document any changes you make, either by appropriate comments or changes in function behavior in the docstrings. 

3. Submit your contribution:

   - Push your changes back to your fork on Github:

     ```bash
     git push origin method-instrumental-variable
     ```

   - Go to Github. The new branch will show up with a green Pull Request button. Make sure the title, message are clear and concise. Verify if the Pull Resquest template is followed in the process.  

## Style

### Coding

We follow the basic python style PEP-8 (https://www.python.org/dev/peps/pep-0008/) for general code and PEP-257 (https://www.python.org/dev/peps/pep-0257/) for docstrings. 

Use one line docstrings for simple functions that do not require further reading to understand. 

```python
def add(a, b):
"""Returns the sum of a and b."""
    return a + b	
```

Follow this guideline when writing one line docstring:

- Triple quotes should start and end in the same line; 
- No blank lines before or after the docstring;
- Give a one phrase summary of the function command ("Do this", "return that");
- If the nature of the returns can not be inferred from code, then add the appropriate type. For instance: """Do X and return a list."""

A detailed docstring should be given for functions that are not literal, usually those functions that a reader could not find out by just reading the code. These mult-line docstrings should provide a summary line, as the one-line dosctring and be followed by a complete description of the parameters and the returns.  The following example illustrates how a multi-line docstring should be formatted. 

```python
def function(a, b, c, opt=0):
"""
	Brief description of what the function does and what it is returned.
	
	Parameters
	----------
	a : type
		description of parameter a
	b : type
		description of parameter b
	c : type
		description of parameter c
	opt: type, optional
		description of parameter opt
	Returns
	-------
	d : type
		description of parameter d
	e : type
		description of parameter e
"""
```

### Git

We adopt the following conventions for branch names in git: use hyphen as a separator. To create a branch with a new feature use a general description followed by the feature.  For example, to implement that `method-instrumental-variable`. 

