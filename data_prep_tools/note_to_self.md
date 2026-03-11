# when creating a new module with hydra
Typically create a sub-folder in the conf folder and keep the yaml file associated with task there. In the config.yaml create the dict entry with the name of subfolder as key and value as the name of the yaml file. Your code can be in src. In main_subcomponent.py use the hydra header and pass paths to conf folder and configs and pass the config object to the function in the code under src. You will have to import the function from src to be able to do this.

Presently creating multiple main files for varying runs. Will subsequently create a full one or divide data_prep and model to separate libraries

All VS code runs cannot use the run button above when working with hydra.. use python main_filenmae.py with hydra inputs in terminal opened in vscode. Eg: python main_input_prep.py or  python main_input_prep.py input.save_gee_data=true input.time_indx=1

# For multiple runs using hydra
for multiple runs, config param= options separated by comma. If its a list, keep in []. This creates a folder called multirun with the combinations of runs instead of creating the outputs in the output folder. For eg
python main_model_train.py model.hidden_layer_size=[32,32],[64] model.activation="relu","logistic" --multirun 

python main_simplified_modeltest.py -m model.act="tanh","gelu","selu","swish","sigmoid" model.solver="adamW"

I have also created an input in config.yaml to ensure the subdirectories created for multirun reflect the combination of the overrides

# Misc things to note about incomplete tasks
In cfg files xxxx is used to indicate caveats or runs to be implemented. No indication of incomplete bits in the py files in src or main

# Quick steps for met data creation:
1. update yaml
2. Run year by year with hydra command: python main_input_prep.py -m input.save_gee_data=false input.time_indx=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22
3. 






