# Applications of Machine Learning for the prediction of drug target interactions in the kinome
Applying Deep Learning and Multi-Task Learning to predict biological target profiles for drug candidates. A short project I pursued within SABS CDT, supervised by Garrett Morris (Dept. of Statistics) and Thierry Hanser (Lhasa Ltd). Here is a quick documentation for my scripts/data/results and how to use them.

**Data :** The files _Interactions_Trainset.tab_, _Interactions_Validset.tab_ and _Interactions_Testset.tab_ contain the interactions used for training, validating and testing respectively. The test-set was formed as a roughly balanced selection between active and inactive compounds. More about the data can be found in my [report](https://github.com/giorkala/Drug-Target-Interactions/blob/master/Presentations/document.pdf). All these files contain triplets as _target-ID, compound-ID, pIC50_, one line per interaction, and have occured after processing _interactions_kinases_all.tab_ which contains all the interactions fetched for 797 kinases. 

Most of the computational experiments conducted for this project are implemented in the following **Python** notebooks:

  1. `Project_SingleTL_CrossVal` : This jupyter notebook compares four single-task methods -- RF, Lasso regression and two NN approaches -- for regressing bioactivity data, with one model per target. RF, LR and the first NN are deployed from Scikit-learn and the second NN (lebelled as `myNN`) is implemented with Keras.

  2. `Project_MTL_CrossVal` : This is similar to the previous notebook but from a multi-task perspective, where we train one model for all the 110 targets. Parameter and architecture selection is conducted with cross-validation. We also explore the use of dropout in a MTL NN. Dropout is useful as it can make the model probabilistic and thus offer the option to produce confidence levels for each prediction.

  3. `Project_Benchmarking` : A notebook that applies and compares all six methods on the validation set and measures time complexity. There's no training here and pre-trained models need to be loaded. 

  4. `Project_Self-training` : The application, and comparison, of two methods that are able to self-train for regression.

  5. `Project_Final_Test` : This is used for the final evaluation of our project. It compares RF with MTL on their ability to impute new values. *Important notice: The file containing fingerprints for every compound is needed here but not included in the git-repo due to size limitations.*

**Demos :** is a folder with html versions of some of the *.ipynb* for presenting the pipeline+results quickly. 

For all the above, the only input we used as features are the **ECFP4**, calculated by RDkit. We also ran a few experiments using feature-based fingerprints (FCFP) and the corresponding materal is under the **FCFP** folder. The rest of the folders are less important and description is skipped at this point.

For more details as well as **references** please have a look at the [report](https://github.com/giorkala/Drug-Target-Interactions/blob/master/Presentations/document.pdf) (which is still in-progress!).
