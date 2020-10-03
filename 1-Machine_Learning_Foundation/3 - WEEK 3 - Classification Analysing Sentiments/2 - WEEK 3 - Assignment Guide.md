# Analyzing product sentiment
In this module, we focused on classifiers, applying them to analyzing product sentiment, and understanding the types of errors a classifier makes. We also built an exciting Jupyter notebook for analyzing the sentiment of real product reviews.

In this assignment, we are going to explore this application further, training a sentiment analysis model using a set of key polarizing words, verify the weights learned to each of these words, and compare the results of this simpler classifier with those of the one using all of the words. These techniques will be a core component in your capstone project.


## Learning outcomes
> - Execute sentiment analysis code with the Jupyter notebook
> - Load and transform real, text data
> - Using the .apply() function to create new columns (features) for our model
> - Compare results of two models, one using all words and the other using a subset of the words
> - Compare learned models with majority class prediction
> - Examine the predictions of a sentiment model
> - Build a sentiment analysis model using a classifier

Download the data and starter code
Before getting started, you will need to download the dataset and the starter Jupyter notebook that we used in the module.

Download the product review dataset here in SFrame format:
[Resouces](https://github.com/Zuyuf/Machine_Learning_Specialization/blob/main/1-Machine_Learning_Foundation/3%20-%20WEEK%203%20-%20Classification%20Analysing%20Sentiments/00%20-%20Resouces.md)

Save both of these files in the same directory (where you are calling Jupyter notebook from) and unzip the data file. Not sure where to save the files? See this guide.
Now you are ready to get started!



## What you will do
> Now you are ready! We are going do four tasks in this assignment. There are several results you need to gather along the way to enter into the quiz after this reading.

> In the Jupyter notebook above, we used the word counts for all words in the reviews to train the sentiment classifier model. Now, we are going to follow a similar path, but only use this subset of the words:

> Often, ML practitioners will throw out words they consider “unimportant” before training their model. This procedure can often be helpful in terms of accuracy. Here, we are going to throw out all words except for the very few above. Using so few words in our model will hurt our accuracy, but help us interpret what our classifier is doing.

> - Use .apply() to build a new feature with the counts for each of the selected_words: 
> > In the notebook above, we created a column ‘word_count’ with the word counts for each review. 
>
> - Our first task is to create a new column in the products SFrame with the counts for each selected_word above, and, in the process, we will see how the method .apply() can be used to create new columns in our data (our features) and how to use a Python function, which is an extremely useful concept to grasp!
>
> - Our first goal is to create a column products[‘awesome’] where each row contains the number of times the word ‘awesome’ showed up in the review for the corresponding product, and 0 if the review didn’t show up. One way to do this is to look at the each row ‘word_count’ column and follow this logic:
> > - If ‘awesome’ shows up in the word counts for a particular product (row of the products SFrame), then we know how often ‘awesome’ appeared in the review,
> > - if ‘awesome’ doesn’t appear in the word counts, then it didn’t appear in the review, and we should set the count for ‘awesome’ to 0 in this review.
>
> We could use a for loop to iterate this logic for each row of the products SFrame, but this approach would be really slow, because the SFrame is not optimized for this being accessed with a for loop. Instead, we will use the .apply() method to iterate the the logic above for each row of the products[‘word_count’] column (which, since it’s a single column, has type SArray). Read about using the .apply() method on an SArray here.


## We are now ready to create our new columns:
- First. you will use a Python function to define the logic above. You will write a function called awesome_count which takes in the word counts and returns the number of times ‘awesome’ appears in the reviews. A few tips:
> i. Each entry of the ‘word_count’ column is of Python type dictionary.

> ii. If you have a dictionary called dict, you can access a field in the dictionary using:
but only if ‘awesome’ is one of the fields in the dictionary, otherwise you will get a nasty error.

> iii. In Python, to test if a dictionary has a particular field, you can simply write:


- Second. Create a new sentiment analysis model using only the selected_words as features: In the Jupyter Notebook above, we used word counts for all words as features for our sentiment classifier. Now, you are just going to use the selected_words

> Use the same train/test split as in the Jupyter Notebook

> Train a logistic regression classifier (use turicreate.logistic_classifier.create) using just the selected_words. Hint: you can use this parameter in the .create() call to specify the features used to be exactly the new columns you just created:

> Call your new model: selected_words_model.

You will now examine the weights the learned classifier assigned to each of the 11 words in selected_words and gain intuition as to what the ML algorithm did for your data using these features. In Turi Create, a learned model, such as the selected_words_model, has a field 'coefficients', which lets you look at the learned coefficients. 

- Third. Comparing the accuracy of different sentiment analysis model: Using the method
> What is the accuracy of the selected_words_model on the test_data? What was the accuracy of the sentiment_model that we learned using all the word counts in the Jupyter Notebook above from the lectures? What is the accuracy majority class classifier on this task? How do you compare the different learned models with the baseline approach where we are just predicting the majority class? Save these results to answer the quiz at the end.


- Fourth. Interpreting the difference in performance between the models: To understand why the model with all word counts performs better than the one with only the selected_words, we will now examine the reviews for a particular product.
