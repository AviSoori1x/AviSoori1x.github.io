
# Building a Natural Language Processing API with SAS Viya, FastAPI, and Docker

<iframe width="100%" height="300" src="https://www.youtube.com/embed/OQuFlayCdh8" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
_Interactively testing the service we will build in this blog_

The notebook for training the topic model and obtaining sentiments is at: [notebook](https://github.com/AviSoori1x/Building_a_SAS_NLP_API_with_FastAPI/blob/main/Building_a_SAS_Viya_NLP_API_using_FastAPI.ipynb)

The code for the FastAPI service and Docker file are at: [service code](https://github.com/AviSoori1x/Building_a_SAS_NLP_API_with_FastAPI)
        

Making sense of text data is a challenge that businesses have faced for decades. Even before the dawn of the machine learning age, there was a significant body of work dedicated to extracting some insights from structured data at scale, but that process for text has constantly lagged behind. With machine learning and associated developments in natural language processing, there has been a transformation in the industry with respect to understanding vast volumes of documents. Needless to say, SAS Viya comes with a variety of analytical capabilities to make this process easier.

SAS Viya has two sets of APIs i.e. lower level CAS APIs for directly calling CAS actions that represent very granular machine learning/ analytics functionaity and SAS Viya APIs that offer a higher level of abstraction. These two levels of abstraction open up a world of possibilities to creative developer teams who want to create enterprise ML/AI Apps for internal or external users.

Here I have used FastAPI and Docker to build a document exploration API that combines several CAS actions. More specifically, with a single HTTP request to this service, an end application would receive the topic distribution in a corpus (NLP speak for body of text) and the sentiment scores, with two levels of granularity. 

I know what you're thinking,what are these topics and why would we even need this? 

The topics I refer to here are semantic grouping of separate documents (NLP speak for pieces of text: think strings) using unsupervised machine learning methods i.e. clusters of similar documents. The measures of semantic similarity are calculated by different algorithms differently. SAS Viya provides Latent Semantic Analysis (LSA) in the SAS Visual Text Analytics Topic node and in the Text Mining Action Set. LSA uses Singular Value Decomposition of the document-term matrix to yield topic weights. SAS Viya also provides Latent Dirichlet Allocation (LDA) in the LDA Topic Modeling Action Set. LDA is a probabilistic approach to topic modeling and is more compute intensive/ relatively slower compared to LSA (which is basically factorizing a big matrix, something computers are pretty efficient at).

The levels of verbosity/ granularity in this context is a reference to the amount of information a request to the service returns. Believe it or not, developers don't want all the pieces of results an algorithms spits out. So it's important to design APIs to give the developer the freedom to chose. Here for the sake of simplicity, you can pass a verbosity flag that will indicate whether to return the sentiment and the topic assignment on a document by document basis or just return aggregate values for the entire corpus.

Note: To follow this blog from begining to end, you must have SAS Viya 3.4 or later and SAS Visual Data Mining and Machine Learning 8.3 or later available in your SAS deployment.

# Let's Deconstruct the Code

I have provided a comprehensive walkthrough of using the Text Mining Action Set and Sentiment Analysis Action in the jupyter notebook at https://github.com/AviSoori1x/Building_a_SAS_NLP_API_with_FastAPI/blob/main/Building_a_SAS_Viya_NLP_API_using_FastAPI.ipynb . Please do go through it, as it is very well documented. We will concentrate on the building the service that will serve the API endpoint here.


The folder structure of the project will be as follows


```python
#-nlpApp
# ├── Dockerfile
# ├── app
# │   ├── _config.py
# │   └── app.py
# ├── nlpService.ipynb
# ├── requirements.txt

```

Feel free to download/clone the repository at the linke or code along. Once you have the higher level nlpApp directory and the app subdirectory, create config.py with your credentials. Since this is a toy example it will be somewhat acceptable but for production use a .env file instead or better yet, if on Azure, use Azure key vault or the equivalent on your prefered cloud. A good example on how to do this with dot-env is given by Miguel Grinberg at https://blog.miguelgrinberg.com/post/the-flask-mega-tutorial-part-xv-a-better-application-structure. I repeat, do not put your credentials out in the open!


```python
#_config.py
def login():
    username = 'enter_your_username'
    password = 'enter_your_password'
    hostname = 'enter_your_hostname'
    return hostname, username, password 
```

Let's take the app.py file apart, this is where all the action happens

First we import the modules we need. This includes FastAPI, the web framwork that facilitates the API serving and of course, SWAT the Scripting Wrapper for Analytics Tranfer which allows us to directly call CAS actions using Python syntax (There's a version for R too!).


```python
#Import the necessary packages and modules.
from fastapi  import FastAPI, File, Query
from pydantic import BaseModel 
from swat import CAS, options
import pandas as pd
import numpy as np
import json
from typing import List
from collections import Counter 
from _config import login 
```

Notice how I import the login function from _config.py. Now use those credentials (again, if in production use a more secure method!)
and connect to CAS, the in-memory highly parallel compute engine that powers Viya. Then load the action sets required (groupings of related analytical functions which are called CAS actions)


```python
host_name, user_name, pass_word = login()
#If pre-Viya 2020
s = CAS(hostname=host_name, protocol='cas', 
            username=user_name, password=pass_word)
#if Viya 2020.x
host_name = host_name + '/cas-shared-default-http/'
port_number = 4443
s = CAS(host_name, port_number, username=user_name,  password=pass_word)

s.loadActionSet(actionSet="textMining")
s.loadActionSet(actionSet="sentimentAnalysis")
s.loadactionset('table')
```

Next define a function to convert a given list of strings to a CASTable with a unique ID and a text column as required by the text analytics CAS actions


```python
def list_to_castable(text_list, table_name):
    docid = list(range(len(text_list)))
    senti_dict = {'docid' : docid,
                'text' : text_list}
    textpd = pd.DataFrame.from_dict(senti_dict)
    print(textpd)
    s.upload(textpd,casout={'name' : table_name, 'caslib' : 'public','replace' : True})
```

Subsequently define a function to generate sentiment scores from text in an in memory CAS Table and returns a cleaned up pandas dataframe with a unique ID column, the sentiment class and the text itself. We pass in the name of the table we want to subject to this action and the number of rows in that table we want to process


```python
def get_sentiments(table_name, total_rows):
    s.sentimentAnalysis.applySent( casOut={"name":"out_sent", "replace":True}, 
                                docId="docid",
                                #Name of the table
                                table={"name":table_name,'caslib':'public'},
                                text="text" )
    #total_rows is the number of rows we want to process
    result = s.table.fetch(table={"name":"out_sent"},maxRows =total_rows,  to=total_rows)  
    text = s.table.fetch(table={"name":table_name,'caslib':'public'},maxRows =total_rows,  to=total_rows) 
    text_list = pd.DataFrame(dict(text)['Fetch']).text.to_list()
    result_pd = pd.DataFrame(dict(result)['Fetch'])
    result_pd['text'] = text_list
    result_pd['uid'] = [i for i in range(total_rows)]
    result_pd = result_pd[['_sentiment_','uid', 'text']]
    return result_pd
```

Then define a function to perform topic modeling utilizing the aforementioned LSA (Latent Semantic Analysis) which functions by performing Singular Value Decomposition on the Document-Term matrix. Here, the previously defined table_name is passed in to the tmMine CAS action in the textMining action set. The number of topics is assigned to the num_topics variable. For reach piece of text, the topic with the highest weight is assigned as the relevant topic. Note that the analytics that requres significant compute is handled in the CAS server and all the post processing is performed in the python runtime. Finally a neat little pandas dataframe with a unique id, the text and the topic with the highest weight is returned.


```python
#A function to generate topic scores from text in an in memory CAS Table
def get_topics(table_name, total_rows, num_topics):
    num_topics = num_topics+1
    s.textMining.tmMine(docId="docid",                                          
    docPro={"name":"docpro", "replace":True, 'caslib' : 'public'},
    documents={"name":table_name, 'caslib' : 'public'},
    k=num_topics,
    nounGroups=False,
    numLabels=5,
    offset={"name":"offset", "replace":True, 'caslib' : 'public'},
    parent={"name":"parent", "replace":True, 'caslib' : 'public'},
    parseConfig={"name":"config", "replace":True, 'caslib' : 'public'},
    reduce=2,
    tagging=True,
    terms={"name":"terms", "replace":True, 'caslib' : 'public'},
    text="text",
    topicDecision=True,
    topics={"name":"topics", "replace":True, 'caslib' : 'public'},
    u={"name":"svdu", "replace":True, 'caslib' : 'public'}  ) 
    topics = s.table.fetch(table={ "name":"topics", 'caslib' : 'public'},maxRows =total_rows,  to=total_rows)
    topic_pd = pd.DataFrame(dict(topics)['Fetch'])
    topic_ids= topic_pd['_TopicId_'].to_list()
    Names = topic_pd['_Name_'].to_list()
    topic_map = {}
    i= 0
    for topic in topic_ids:
        topic_map[str(topic)] = Names[i]
        i+=1
        
    #Assigning each document to the most heavily weighted topic that the process discovered    
    docpro = s.table.fetch(table={ "name":"docpro", 'caslib' : 'public'},maxRows =total_rows,  to=total_rows)  
    docpro = pd.DataFrame(dict(docpro)['Fetch'])
    topics = []
    for i in range(total_rows):
        topics.append(topic_map[docpro.iloc[i][1:num_topics].idxmax().strip('_Col')+'.0'])
    text = s.table.fetch(table={"name":table_name,'caslib':'public'},maxRows =total_rows,  to=total_rows) 
    textpd = pd.DataFrame(dict(text)['Fetch'])
    textpd['topics']= topics
    textpd['uid'] = [i for i in range(total_rows)]
    textpd = textpd[['uid','text','topics']]
    return textpd
```

We then initiate the API as follows. The metadata indicating parameters title, description and version are optional but I encourage you to add them for documentation and ease of maintenability, which go hand in hand.


```python
# initiate API
app = FastAPI(
    title="SAS Document Explorer",
    description="An NLP API that leverages lower level SAS Viya Text Analytics APIs to perform sentiment classification as well as topic modeling",
    version="0.1")

```

Then we use pydantic's Basemodel to define a model for the post request. I choose to include the sole paramter I will pass in the request body i.e. text_list which is a list of strings


```python
class InputListParams(BaseModel):
    text_list: List[str] = Query(None)
```

Then we define the post request as follows:


```python
@app.post('/analyze_text')
def analyze_text(num_topics: int, total_rows: int, table_name: str, verbose: int,params:InputListParams):
    if len(params.text_list)>= total_rows:
        total_rows == total_rows 
    else:
        total_rows = len(params.text_list)
    cas_upload = list_to_castable(params.text_list, table_name)
    sentiments = get_sentiments(table_name, total_rows)
    topics = get_topics(table_name, total_rows, num_topics)
    result = pd.merge(sentiments, topics[['uid','topics']], on='uid', how='left').drop('uid', axis=1)
    #reordering columns 
    result = result[['text','_sentiment_', 'topics']]
    json_string = result.to_json()
    #Give aggregations of sentoiments and topic counts for convenience
    jsonified_result  = json.loads(json_string)
    sentiment_agg = dict(Counter(result._sentiment_.to_list()))
    topic_agg = dict(Counter(result.topics.to_list()))
    jsonified_result['sentiment_agg'] = sentiment_agg
    jsonified_result['topic_agg'] = topic_agg
    if verbose == 0: 
        agg_data = {}
        agg_data['sentiment_agg'] = sentiment_agg
        agg_data['topic_agg'] = topic_agg
        return agg_data
    return jsonified_result
```

FastAPI uses type hints and is quite useful in defining the API parameters. Note that each parameter, including the one defined above indicates the type of the argument. I have written some logic to ensure that the total_rows is at less than or equal to length of the list of strings or else, assign the length of the list of strings as the maximum number of rows for analysis.
I call the functions I've defined above to get the sentiment and most relevant topic and join the tables to give a consolidated analyses on each string, with respect to sentiment and theme. Then I add a feature that gives the developer the freedome to define the level of detail desired. i.e. verbosity. If verbosity is 1, a call to this end point will result in a verbose return with sentiment and relevant topic for each string. If verbosity is 0, only the aggregate number of tweets belonging to each sentiment classification and topic will be returned. i.e. positive, neutral and negative and sum of tweets for each topic.

Now you've gone through the entire .py file with the API logic. For you to test out the API, I recommend building the container image and running the container. For that you need the Dockerfile, which I have provided in my repository. It reads as follows:


```python
FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY ./app /app

CMD ["uvicorn", "app:app","--reload", "--host", "0.0.0.0", "--port", "80"]
```

Note that we build the container from the base image provided by the creator of FastAPI(He's done some really nice work with Typer too, check it out!). The COPY instruction copies (Docker is pretty straightforward, no cryptic commands) from the source i.e. the directory structure you probably got on your system when you downloaded the files from the repository, to the container file system. This way we copy requirement.txt which lists all the depencies to run FastAPI, connect to CAS(with SWAT) etc. and install them in the container with the RUN instruction. Then we copy the app folder contents to the container file system and finally run the command to start FastAPI

Now that you understand what all these lines are in the Dockerfile, let's build the image and start up the container, Finally! First run(at the command line):


```python
docker build -t myimage .
```

You don't have to call it myimage but that's what I chose. Then start up the container:


```python
docker run -d --name mycontainer -p 80:80 myimage
```

I really recommend that you download Docker Desktop which allows you to peer inside the container you just spun up. FastAPI is nice in that it comes with automatic documentation. You can test our deployed service with interactive API documentation using the Swagger UI at http://127.0.0.1:80/docs. You can also check out the redoc based UI at http://127.0.0.1:80/redoc

Also please note that the list of strings passed int he body of the request should be devoid of any escape characters and should be enclosed in double quotation marks to be conformant with json formatting. For your convenience, use the following python code to create this list of strings from the text column of your dataframe. In this snippet, the dataframe is assigned to a variable 'df' and the text column in 'df' is 'text'.


```python
strs = df.text.to_list()[:1000]
strs = ["'{}'".format(str(item).replace(""", " ")) for item in strs]
```

In this blog post, you've built your own higher level API to interact with more granular text analytics APIs in SAS Viya, containerized it with Docker, and deployed it locally. The next step would be to deploy this to a cloud service for serverless container deployment like AWS Fargate, Google Cloud Run, or Azure Container Instance. This will be discussed in a future blog post.
