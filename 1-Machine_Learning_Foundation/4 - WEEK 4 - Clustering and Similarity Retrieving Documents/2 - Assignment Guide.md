# Retrieving Wikipedia articles
In this module, we focused on using nearest neighbors and clustering to retrieve documents that interest users, by analyzing their text. We explored two document representations: word counts and TF-IDF. We also built an Jupyter notebook for retrieving articles from Wikipedia about famous people.

In this assignment, we are going to dig deeper into this application, explore the retrieval results for various famous people, and familiarize ourselves with the code needed to build a retrieval system. These techniques will be key to building the intelligent application in your capstone project.

Follow the rest of the instructions on this page to complete your program. When you are done, instead of uploading your code, you will answer a series of quiz questions (see the quiz after this reading) to document your completion of this assignment. The instructions will indicate what data to collect for answering the quiz.

## Learning outcomes
> - Execute document retrieval code with the Jupyter notebook
> - Load and transform real, text data
> - Compare results with word counts and TF-IDF
> - Set the distance function in the retrieval
> - Build a document retrieval model using nearest neighbor search
> - Resources you will need
> - You will need to install the software tools described in the Module 1 reading. Instructions are provided here.

Download the data and starter code
Before getting started, you will need to download the dataset and the starter Jupyter notebook that we used in the module.

[Resources](https://github.com/Zuyuf/Machine_Learning_Specialization/blob/main/1-Machine_Learning_Foundation/4%20-%20WEEK%204%20-%20Clustering%20and%20Similarity%20Retrieving%20Documents/00%20-%20Resouces.md)

Save both of these files in the same directory (where you are calling Jupyter notebook from) and unzip the data file. Not sure where to save the files? See this guide.
Now you are ready to get started!


### What you will do
Now you are ready! We are going do three tasks in this assignment. There are several results you need to gather along the way to enter into the quiz after this reading.

## Compare top words according to word counts to TF-IDF:
> In the notebook we covered in the module, we explored two document representations: word counts and TF-IDF. 
> Now, take a particular famous person, 'Elton John'. What are the 3 words in his articles with highest word counts?
> What are the 3 words in his articles with highest TF-IDF?
> These results illustrate why TF-IDF is useful for finding important words. Save these results to answer the quiz at the end.


## Measuring distance: 
> lton John is a famous singer; let’s compute the distance between his article and those of two other famous singers. 
> In this assignment, you will use the cosine distance, which one measure of similarity between vectors, similar to the one discussed in the lectures. 
> You can compute this distance using the turicreate.distances.cosine function. 
> - What’s the cosine distance between the articles on ‘Elton John’ and ‘Victoria Beckham’? 
> - What’s the cosine distance between the articles on ‘Elton John’ and Paul McCartney’? 
> - Which one of the two is closest to Elton John? Does this result make sense to you? Save these results to answer the quiz at the end.



## Building nearest neighbors models with different input features and setting the distance metric:
> In the sample notebook, we built a nearest neighbors model for retrieving articles using TF-IDF as features and using the default setting in the construction of the nearest neighbors model. Now, you will build two nearest neighbors models:
> - Using word counts as features
> - Using TF-IDF as features
In both of these models, we are going to set the distance function to cosine similarity. Here is how: when you call the function


> - What’s the most similar article, other than itself, to the one on ‘Elton John’ using word count features?
> - What’s the most similar article, other than itself, to the one on ‘Elton John’ using TF-IDF features?
> - What’s the most similar article, other than itself, to the one on ‘Victoria Beckham’ using word count features?
> - What’s the most similar article, other than itself, to the one on ‘Victoria Beckham’ using TF-IDF features?
> Save these results to answer the quiz at the end.

