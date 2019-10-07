# Drug-Target-Interactions
Applying Deep Learning and Multi-Task Learning to predict biological target profiles for drug candidates. A short project I pursued within SABS CDT, supervised by Garrett Morris (Dept. of Statistics) and Thierry Hanser (Lhasa Ltd). Here is a quick documentation for my scripts/data/results and how to use them, descibed in descending order of importance (most likely).

**Data :** The files _Interactions_Trainset.tab_, _Interactions_Validset.tab_ and _Interactions_Testset.tab_ contain the interactions used for training, validating and testing respectively. The test-set was formed as a roughly balanced selection between active and inactive compounds. More about the data can be found in my report.

**Project_CrossVal_SingleTL :** This jupyter notebook compares four single-task methods -- RF, Lasso regression and two NN approaches -- for regressing bioactivity data, with one model per target. RF, LR and the first NN are deployed from Scikit-learn and the second NN is implemented with Keras.

**Project_MTL :** This is similar to the previous notebook but from a multi-task perspective, where we train one model for all the 110 targets. Parameters are selected with grid-search after spliting the train set to 75-25%.

**Project_MTL-Dropout :** This notebook expands the previous by exploring the use of dropout in a MTL NN. Dropout is useful as it can make the model probabilistic and thus offer the option to produce confidence levels for each prediction.

**Project_Self-training :** The application, and comparison, of two methods that are able to self-train for regression.

**Project_Final_Test :** This is the notebook used for the final evaluation of our project. It compares RF with MTL on their ability to impute new values. 
