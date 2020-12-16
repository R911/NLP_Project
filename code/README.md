This is the main entry point into the code. From here there are sub-folders for the specific takss and jobs.

The data input, and embeddings classes can be found in data_input

The code to produce metrics and TSNEs can be found in evaluation

The code to produce the averages and std of the results can be found in graphstats

The code for LINE can be found in LINE

The code for node2vec can e found in node2vec

There are some helpful utility functions in the Utils foler as well as the utils.py file.

The code is run from main, an example pf a run is provided below where the code is run with Line on the flicker dataset for 5 iterations.

python3 main.py -m Line -d Flickr -r 5

There is also a batch scripts folder that contains the scripts used to run the code on the discovery cluster. These files must be copied out into the code directory for them to work. 
