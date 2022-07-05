# Running a flower classification example with Mouffet

A full documentation explaining in detail the example can be found here:
https://mouffet.readthedocs.io/en/latest/tutorial.html

  
## Install Mouffet

Details instructions to install Mouffet can be found on the main page of the Mouffet github repository:
https://github.com/Vin985/mouffet

  
## Install additional dependencies

Additional dependencies are required to run the example. To install them, in your command line type

    pip install tensorflow tensorflow-datasets plotnine sklearn

  
## Download the exemple

Download the example at this address: 

    https://github.com/Vin985/mouffet/blob/main/examples/flowers_example.zip

Extract the archive where you desire. Open a terminal and go to the 'flowers' folder of the example.

  
## Launch the training example

To run the training example, you only need to type this in the command line

    python training.py

By default, you will find the models saved in the results/models folder

  
## Launch the evaluation example

To run the training example, you only need to type this in the command line

    python evaluation.py

Note: Do not worry about the warning, this is a result about creating a 'fake' class when using
thresholds.

The evaluation will generate the following files:
 - In the results/predictions are the predictions generated on the test dataset. Data is saved in the
 feather format ()
 - In the results/evaluations are the results of the evaluation. Results are saved in a folder
 named after the day in the DDMMYYYY format. Inside, there will be a csv file containing the evaluation
 metrics as well as two pdf. One contains all confusion matrices for each evaluation. The other
 contains the plots of f1_score in function of accuracy for all results. All fixes are prefixed by
 a timestamp in the hhmmss format.

  
## Launch a full run

To launch a run that will perform both training and evaluation, we need to call the file run.py with
the name of the run we want to perform. This name is actually the name of the folder that contains
the configuration files. Inside the run.py file, we told the RunHandler to look inside the "config"
folder to look for run folders (via the 'run_dir' option). Here we put all configuration files inside
the a flowers folder. Therefore, in the command line, we need to type

    python run.py flowers

The results for a run are the same as when training separately. The only difference is that results
are saved in a different folder specific for each run. By default, results are saved in the
results/runs under a subfolder with the run name. Therefore all results here can be found under the
results/runs/flowers folder.