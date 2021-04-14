# Interactive Explainable Machine Learning with SAS Viya, Streamlit, and Docker

If you want to get to the part about building the app and the code walk-through, skip the first 3 paragraphs ;). But in all seriousness, I welcome you to read the introduction, just so you understand why this is important.

                                                                   ----
Whether you notice at first glance or not, machine learning based functionalities have become a core component of numerous digital products you regularly consume. Similarly, these often complex algorithms have made their way into business processes in many organizations that have a huge impact on your life. In my experience alone, I have seen algorithms as complex as gradient boosting and deep neural networks in production at financial institutions and retailers, tightly integrated with business processes. In financial services in particular, a prediction from a model could be part of a loan or credit card approval process. In both cases, the output from these models have a significant impact on people’s lives.

Given this widespread adoption, for regulatory purposes and to ensure that customers get the best possible experience, it’s important to be able to explain why a certain prediction was made for a given set of input features. In other words, for an observation, how did the values of each of the features affect the predicted value? So if you’re a lender you’d better be able to explain why Bob’s loan was approved and John’s wasn’t!

When it comes to local interpretability (explaining each prediction) LIME (Local Interpretable Model-Agnostic Explanations) and several variants of SHAP are the most commonly used algorithms. I won’t go into how each of these work in this article. Basically, LIME works by approximating a complex model by a series of simple models around each observation and SHAP (Shapley values) is based on a game theoretic approach. Good news is that Visual Data Mining and Machine Learning (VDMML) on SAS Viya 3.x and later have multiple variants of both these algorithms for you to use out of the box!

In this blog post, I will use SAS SWAT package for Python to train a gradient boosting model in SAS Viya, save the model artifact (a SAS astore, a BLOB much like a pickle file), and build a containerized Streamlit app to score new observations against the model and visualize interpretability plots using whichever algorithm you like!

You must have SAS Viya 3.4 or later and SAS Visual Data Mining and Machine Learning 8.3 or later available in your SAS deployment.
The notebook for training the model is at: https://github.com/AviSoori1x/Explainable-ML-with-SAS-Viya/blob/main/SAS_Viya_Explainable_ML.ipynb
The code for the Streamlit app and Docker file are at: https://github.com/AviSoori1x/Explainable-ML-with-SAS-Viya

Feel free to follow along.
Alright let’s get started! My dataset for this walkthrough is titled HMEQ and the goal is to classify whether an individual is likely to default on a home loan or not, based on a number of factors. This outcome is recorded as a binary flag in the field titled BAD (Binary Attribute indicating Default: 0 for no default, 1 for default). This could be downloaded at https://support.sas.com/documentation/onlinedoc/viya/exampledatasets/hmeq.csv . Also, I keep referring to a ‘notebook’ in the data and model training portion as I used a Jupyter notebook here to write my code. Feel free to use any IDE or even the Python interpreter if you prefer. I will switch to Visual Studio code for the Streamlit and Docker portion, but there also, the choice is entirely yours :)

Download this dataset to the same directory as your notebook environment for training the model. Install SWAT if you haven’t already. Then import all the required modules as shown below:

Connect to the CAS server (the in-memory compute engine that powers SAS Viya) and load the relevant action sets. Action sets are related groups of algorithms baked into the CAS runtime, which in turn are called, CAS actions. For example the ‘decisionTree’ action set has multiple actions. Some of these actions are for training and scoring various tree based algorithms such as random forests and gradient boosting models.

Please note that the host, port, username and password are strings that are specific to your environment.
Read the dataset into a pandas DataFrame and inspect it:

Upload this data set into CAS. Let’s call the in memory table ‘hmeqTest’ and place it in the ‘public’ Caslib. Caslib is just a collection of CAS Tables. Also set ‘replace’ to the True Boolean value to replace if a table by the same name already exists.

Since the goal of this blog is not to perfect the model, but to understand how to quickly train a model and build an app around it, we will forgo any attempts at feature engineering. We will use the tuneGradientBoostTree from the autotune action set to train and obtain a gradient boosting model with optimal hyperparameters. Note that the nominal and interval variables have been clearly identified. The resulting final model is saved in a CAS table bearing the name ‘gradboosthmeqtest’.

Once this step is completed, export the trained model’s astore to a new CAS table. Promote the original base table(‘hmeqTest’) and the exported the astore, to global scope. This will make it possible for the new CAS connection created for the Streamlit app to score new data and generate explanations using this model.

Now, before we ‘appify’ things with Streamlit, we can test out our model in the notebook or another Python REPL environment.
Let’s create a sample observation to be scored as a Python dictionary:

Then we create a small helper function to convert the python dictionary with the sample observation to a pandas DataFrame. This could be done with a single line of code but the data types end up changing. Hence we construct this slightly verbose function:

Then we use this helper function to get the pandas DataFrame from the dict and upload it as a CAS table to be scored against the model. Here we create it as a python dictionary, convert it into a pandas DataFrame and then upload it to a CAS Table.

This CAS Table (consisting of just one row of values and the field names) will be aptly named ‘realtime’ in the ‘public’ caslib. To keep things simple, we will be scoring in CAS, but in practice I would much rather use MAS (Micro Analytics Service i.e. SAS’ low latency scoring and decisioning runtime). The result of scoring ‘realtime’ against the model will be stored in a CAS table called ‘realscore’ in the same caslib. We can also examine the score thus generated as indicated in the code.

The results:

Looks like our fictional customer is likely to default on his/her home equity loan (at least according to this model). Now we can pack this code into a neat little scoring function for later use in our app.

Important: Before we use the CAS action for explaining a particular result, we should add the ‘BAD’ field to the observation table and populate it with the ‘I_BAD’ value i.e. the prediction we got from the model. Then we should upload the pandas DataFrame to CAS as the ‘realtime’ CAS table. This is done with the following two lines of code:

Now we can finally get to the part we’ve all been waiting for ( assuming you’ve read so far). We will use the linearExplainer CAS action in the Explain Model action set to get the Shapley values for this particular observation. Take a look at the code and I will walk through each step.

Here, the table is the analytical base table. i.e.the training data. The query is the CAS table with the observation we want to get the Shapley values for (remember: we just added the predicted BAD value to it). Model Table is the CAS table that we exported and promoted to the global scope. Model table type in this case is our beloved astore. There are other score code types such as DS2 (Data Step 2) in the SAS ecosystem but we need not elaborate on that now. As you saw in the score result, this binary classifier gives us 2 probability values. I chose to have the explainability algorithm consider the Probability of Bad being 1, i.e. P_BAD1 as the target.

Also note that the target column in the original dataset ‘BAD’ is also present in the input list. Let’s take a look at these SHAP values:

Here, the intercept could be interpreted as a bias, i.e. a value used to adjust the local linear model used to explain this observation. If the intercept dominates the other values in terms of magnitude, then no given input feature has a dominating impact on a given result. This is not the case in the above example, as Loan amount has a Shapley value of 0.338, which is twice the intercept.
We can visualize this using any data visualization library in the Python ecosystem. Let’s do so with Altair

Now we move onto the fun part. Wrapping this up as an app with an interactive UI and making sure it’s portable and deployable. This could be done fairly easily with Streamlit and Docker.

Why Streamlit? Streamlit lets you build a UI for any API using pure Python. All you have to do is install streamlit in your environment, import Streamlit at the top of your script and start declaring UI components. I have added comments line by line describing exactly what I’m doing and why I’m doing it.

The entire logic of the app is all present in a single file with the .py extension. Let’s call this file app.py. I welcome you to go to the source at https://github.com/AviSoori1x/Explainable-ML-with-SAS-Viya , read the instructions and get the app running locally.

The instructions are as follows.
Download the files from the Github to your machine. The folder structure should look like this:

Then execute the following instructions.

1. Run the notebook from the first cell to the last (at https://github.com/AviSoori1x/Explainable-ML-with-SAS-Viya/blob/main/SAS_Viya_Explainable_ML.ipynb). If you have been coding along as you were reading, you must have done this by now. This creates and promotes the analytical base table and the final trained model as an astore in CAS. Now go to the streamlitApp directory with all the files. Dockerfile is here.
2. Run the following commands at the command line (I’m assuming you have Docker installed. If not, install Docker!). Your present working directory should be the streamlitApp folder.
    1. First run at the terminal: docker build -f Dockerfile -t app:latest .
    2. Then run: docker run -p 8501:8501 app:latest
    3. Then test out your app at: http://localhost:8501/


Make sure both the model training (notebook) and the connection to the CAS server is to the same host with the permissions to access CAS tables in the global scope.
This is not the most straightforward Streamlit app as we have to persist the CAS connection over the selective execution of functions for scoring and generating explanations. There are several other things we need to persist as well. So I use the wonderful SessionState.py gist (https://gist.github.com/FranzDiebold/898396a6be785d9b5ca6f3706ef9b0bc) created by Thiago Teixeira and modified by Franz Diebold. You will find this in the files you have already downloaded and is visible in the directory tree.
Now you have built a machine learning based application that not only allows you to predict new observations and explanations, but it’s also deployed in a containerized manner which makes it portable. In a future blog post, I will walk you through deploying it on GCP Compute Engine, AWS Fargate and Azure App Engine.
I hope you enjoyed this article and developed an appreciation for making machine learning related explorations more interactive!
