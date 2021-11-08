
/**********/
/* FLAKE8 */
/**********/

Run the "linter" flake8 to detect:
- logical lints: code errors, code with potentially unintended results, dangerous code patterns
- stylistic lints: code not conforming to defined conventions
Config file:
	tox.ini (could also be .flake8 or setup.config)
	output-file does not seem to work => specified at the command
	ignore F403 & F405 for the moment (import everything)

From package directory
$ flake8 --output-file=logs_quantools_flake8.txt
Or from outside the package directory (the config file must be in the same directory)
$ flake8 --output-file=logs_quantools_flake8.txt quantTools


/*********/
/* BLACK */
/*********/

Run black to format code 
Config file:
	pyproject.toml
	remarks: 
	
From repository (where the pyproject.toml file is located)
$ black .
