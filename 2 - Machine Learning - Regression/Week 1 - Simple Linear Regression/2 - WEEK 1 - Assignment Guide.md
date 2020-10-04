# Fitting a simple linear regression model on housing data

## Regression Week 1: Simple Linear Regression Assignment
> Predicting House Prices (One feature)
>
> In this notebook we will use data on house sales in King County, where Seattle is located, to predict house prices using simple (one feature) linear regression. You will:
> - Use SArray and SFrame functions to compute important summary statistics
> - Write a function to compute the Simple Linear Regression weights using the closed form solution
> - Write a function to make predictions of the output given the input feature
> - Turn the regression around to predict the input/feature given the output
> - Compare two different models for predicting house prices

### What you need to download
> [REG01-NB01.ipynb.zip](https://d3c33hcgiwev3.cloudfront.net/00IIFeIoEemx8A5HK6Ls8g_56295025ae1d45ad900e1fd3f77e01c0_REG01-NB01.ipynb.zip?Expires=1601942400&Signature=Y9mQorONfn2-WB0bvnTbAf2Ki2kvBmTBmzAJHnkNJ0BtWZ4RrRkgaCiVQyIFNVZAA-iTdbGrGaQA37TLa4EoXwoJB0hIjGgtaGJW8SkKc58WBLMSIGqSVPfFpbK5TmnPRmrNbhOwFLA61uHexK~jnd5PnMEed5oFQSyww7VrTPc_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A)
>
> If you are using Turi Create:
> - Download the King County House Sales data In SFrame format: [home_data.sframe.zip](https://d3c33hcgiwev3.cloudfront.net/00FsKOIoEemELQpo9cj5Ig_579152614f2b4a399ac90a939360749e_home_data.sframe.zip?Expires=1601942400&Signature=K-MCvaO9jdOfdfKrjDcoFzJs9kMVLe04jz~vN~D9hJoQ-nhyyXdTxU0HHhH0FnkMBsH-Ar~~Rv7hEkGgkCqZsvvQpdu8n7b0JQCxxKjyiNc21Tyb3VKth8Pbio8LCjrgfyuVe0oGXQPpoUiKI2pzTBL7peHswfxkcIgh3tMizPk_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A)
> 
> If you are not using Turi Create:
> - Download the King County House Sales data csv file:  [kc_house_data.csv.zip](https://d3c33hcgiwev3.cloudfront.net/_46994807796a1213d2699c6d9a09667c_kc_house_data.csv.zip?Expires=1601942400&Signature=U-G32t572nm8kQNdSw9PwoknBEmtVy0TnAqPqOzkkMJRltKAcN9wIBp4z54eq5vqDk9EtSBh9L4RilsUJcfOKO7EsgPJXPGefF0h1Nz7AxIFr7YmK-9rGdXKS0L6FeqH3I2v4-SaTc3j-p4R-qH5Zaa-pA-zBAIN-Lbe7H1tbbc_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A)
> - Download the King County House Sales training data csv file:  [kc_house_train_data.csv.zip](https://d3c33hcgiwev3.cloudfront.net/_46994807796a1213d2699c6d9a09667c_kc_house_train_data.csv.zip?Expires=1601942400&Signature=OtuGD5xtS2Dc~n6t2qPMWivYSRK9Fxz-7ugko2yUmfDebtjGbZavyWwHZoVGT72c0v8pyZgHCGdIZLISKFXuorFmgjRmdBmMXQxyxSlgtebXR1Hw2UNRPtsf-yXccZGHbvQBlTAsxNc3VfGl~KSIqb-5WeNgiQ1q4XeBmVmiYxg_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A)
> - Download the King County House Sales testing data csv file:  [kc_house_test_data.csv.zip](https://d3c33hcgiwev3.cloudfront.net/_46994807796a1213d2699c6d9a09667c_kc_house_test_data.csv.zip?Expires=1601942400&Signature=LH6jMqafP1ynHznR76kUBvZJEwpptGmLY5kq9EiNZB0NRUCzsVvm1QDKbhiHnYDVEAfeP1ehY5tfbKQPNDs-Y9ZH3qtrVWiuCNHrwd2CjGpNGIxNVlGJsF~kFp8F~1maG263TjHs7DCS5UpzqCd94L0kGDntVI26j7kOUwy8fVQ_&Key-Pair-Id=APKAJLTNE6QMUY6HBC5A)
>
> **- IMPORTANT: use the following types for columns when importing the csv files. Otherwise, they may not be imported correctly: [str, str, float, float, float, float, int, str, int, int, int, int, int, int, int, int, str, float, float, float, float]. If your tool of choice requires a dictionary of types for importing csv files (e.g. Pandas), use:**
>               ```python
>                 dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float, 'grade':int, 'yr_renovated':int, 'price':float, 'bedrooms':float, 'zipcode':str, 'long':float, 'sqft_lot15':float, 'sqft_living':float, 'floors':str, 'condition':int, 'lat':float, 'date':str, 'sqft_basement':int, 'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int}
>               ```


## Assignment Questions
- **1.** If you are using SFrame, import Turi Create and load in the house data, otherwise you can also download the csv. (Note that we will be using the training and testing csv files provided). e.g in python with SFrames:
            ``` sales = turicreate.SFrame('kc_house_data.gl/') ```

- **2.** Split data into 80% training and 20% test data. Using SFrame, use this command to set the same seed for everyone. e.g. in python with SFrames:
            ``` sales = turicreate.SFrame('kc_house_data.gl/') ```
      For those students not using Turi Create please download the training and testing data csv files.
      From now on we will train the models using train_data. It will be important that we use the same split here to ensure the results are the same.

- **3.** Write a generic function that accepts a column of data (e.g, an SArray) ‘input_feature’ and another column ‘output’ and returns the Simple Linear Regression parameters ‘intercept’ and ‘slope’. Use the closed form solution from lecture to calculate the slope and intercept. e.g. in python:
             ``` def simple_linear_regression(input_feature, output):
                      [your code here]
                  return(intercept, slope)  ```

- **4.** Use your function to calculate the estimated slope and intercept on the training data to predict ‘price’ given ‘sqft_living’. e.g. in python with SFrames using:
              ``` input_feature = train_data[‘sqft_living’]
                  output = train_data[‘price’]
                   ```
        save the value of the slope and intercept for later (you might want to call them e.g. squarfeet_slope, and squarefeet_intercept)

- **5.** Write a function that accepts a column of data ‘input_feature’, the ‘slope’, and the ‘intercept’ you learned, and returns an a column of predictions ‘predicted_output’ for each entry in the input column. e.g. in python:
              ``` def get_regression_predictions(input_feature, intercept, slope)
                      [your code here]
                      return(predicted_output)```

- **6. Quiz Question: Using your Slope and Intercept from (4), What is the predicted price for a house with 2650 sqft?**

- **7.** Write a function that accepts column of data: ‘input_feature’, and ‘output’ and the regression parameters ‘slope’ and ‘intercept’ and outputs the Residual Sum of Squares (RSS). e.g. in python:
               ``` def get_residual_sum_of_squares(input_feature, output, intercept,slope):
                        [your code here]
                        return(RSS)```
Recall that the RSS is the sum of the squares of the prediction errors (difference between output and prediction).

- **8. Quiz Question: According to this function and the slope and intercept from (4) What is the RSS for the simple linear regression using squarefeet to predict prices on TRAINING data?**

- **9.** Note that although we estimated the regression slope and intercept in order to predict the output from the input, since this is a simple linear relationship with only two variables we can invert the linear function to estimate the input given the output!

      Write a function that accept a column of data:‘output’ and the regression parameters ‘slope’ and ‘intercept’ and outputs the column of data: ‘estimated_input’. Do this by solving the linear function output = intercept + slope*input for the ‘input’ variable (i.e. ‘input’ should be on one side of the equals sign by itself). e.g. in python:
              ``` def inverse_regression_predictions(output, intercept, slope):
                      [your code here]
                      return(estimated_input)```

- **10. Quiz Question: According to this function and the regression slope and intercept from (3) what is the estimated square-feet for a house costing $800,000?**

- **11.** Instead of using ‘sqft_living’ to estimate prices we could use ‘bedrooms’ (a count of the number of bedrooms in the house) to estimate prices. Using your function from (3) calculate the Simple Linear Regression slope and intercept for estimating price based on bedrooms. Save this slope and intercept for later (you might want to call them e.g. bedroom_slope, bedroom_intercept).

- **12.** Now that we have 2 different models compute the RSS from BOTH models on TEST data.

- **13. Quiz Question: Which model (square feet or bedrooms) has lowest RSS on TEST data? Think about why this might be the case.**
