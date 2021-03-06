# CIS 531 TERM PROJECT

[https://github.com/jdeweese1/cis-531-ml-project](https://github.com/jdeweese1/cis-531-ml-project)

Install python dependencies. Depending on your environment, they maybe already installed
```
emojis
matplotlib
nltk
numpy
pandas
textstat
scipy
sklearn
vaderSentiment
tweepy
```
You should be able to run a command like `$ pipenv install ` and have your installer tool install from the `Pipfile`. Your milage may vary depending on your machine and local Python environment.

Obtain the [data](https://dataverse.mpi-sws.org/dataset.xhtml?persistentId=doi:10.5072/FK2/ZDTEMN). If you are my TA, then you can also find the relevant data in the comments section of the canvas submission. To run the program, you will need to fix 3 errors that will prompt you to input some variables, specific to your system and data. This would be you input file, and the keys for Twitter. Run the `twitter_extractor.py` program to obtain tweet text and other details.

Run main_ml.py
Inside `main_ml.py` you should see a section in there in the top 60 lines, where configuration variables can be set. This program expects the input file from the last step to be called `outfile.csv` This changes what kind of features the program uses to bulid the models. 
When the program runs, it will print lots of information to the console, and yield multiple warnings (in an ideal world, my model solvers would converge) 
The program will output confusion matrices and classification reports into the `plots` and `classif_reports` directories.

Output directories are generated using the following formatting:
`f'./plots/{data_mode}_{smote_desc}_norm_to_{normalized_to}/'`, where `data_mode` is either `test` or `train` (we run the model against the training data to decide how much it is overfit to the data), `smote_desc` is `[un]smoted`, if this value is `unsmoted`, then the program does not try to balance classes (it chooses test data in a way representative of reality), `normalized_to` indicates how the plot should be normalized: show each box as its percentage of all boxes in the plot, or its percentage in relation to it's row or to it's column.

Filenames for outputs are generated by the following formatting:
 `f'{dir_path}report{mdl_name}_{result_column}_{tfidf_text_colmun_name}_{tag}.EXTENSION'`.
 `mdl_name` is the name of the model class, `result_column` indicates if using 2 or 3 class classificaton. `tfidf_text_colmun_name` provides us with informaton as to which column generated the tfdif input for vocabulary. `tag` is a short description of the model. 

There is also the project_paper directory. It is used to build my report, using `make`. `The main_ml.py` program should be run before creating the report, as the report relies on some files generated by the other program. 

The report's make commands are `paper`, `appendix`, `tex`, `tex2pdf`, and `clean`. pandoc and pandoc-include should be on your system to make the paper.
