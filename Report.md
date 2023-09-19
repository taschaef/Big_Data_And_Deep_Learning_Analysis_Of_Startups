1. Explain the purpose of your analysis 
The purpose of my analysis was to create a tool that can help it select the applicants for funding with the best chance of success in their ventures. I used machine learning and neural networks in the provided dataset to create a binary classifier that can predict whether applicants will be successful if funded by Alphabet Soup.

From Alphabet Soup’s business team, I received a CSV containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. Within this dataset are a number of columns that capture metadata about each organization, such as:

EIN and NAME—Identification columns
APPLICATION_TYPE—Alphabet Soup application type
AFFILIATION—Affiliated sector of industry
CLASSIFICATION—Government organization classification
USE_CASE—Use case for funding
ORGANIZATION—Organization type
STATUS—Active status
INCOME_AMT—Income classification
SPECIAL_CONSIDERATIONS—Special considerations for application
ASK_AMT—Funding amount requested
IS_SUCCESSFUL—Was the money used effectively

Using this dataset, I trained neural network models to predict whether future applicatns would recieve funding or not. 

2. Using bulleted lists and images to support your answers, address the following: 
    - What variable(s) are the target(s) for your model?
        - Whether the organization was funded or not. Variable = is_successful
    - What variable(s) are the features for your model?
        - All other variables, such as application type, affiliation, classification, use case, organization type, status, income amount, specail considerations, and asked-for amount. 
    - What variable(s) should be removed from the input data because they are neither targets nor features?
        - NAME and EIN variables should be removed 

3. Compiling, Training, and Evaluating the Model
    - How many neurons, layers, and activation functions did you select for your neural network model, and why?
        - To begin, I used three dense layers, with the first layer having 100 neurons and a relu activation, the second having 50 neurons and an elu activation function, and the third having only 1 neuron. This set of layers had the output layer utilizing a sigmoid function. I chose these characteristics because they are fairly typical of a neural network. 

![Alt text](image.png)


    - Were you able to achieve the target model performance? 
        - No, I achieved 72.9% accuracy on my first run-though.

![Alt text](image-1.png)

    - What steps did you take in your attempts to increase model performance?
        - I added three more layers to the neural network model with increasing number of nodes. 
        - I tried out different activation functions. On my second attempt I used 'elu', 'sigmoid', and 'relu'. On my third attempt I decided to only use 'relu' and 'sigmoid' to slim down my code. 
        - I tested different amounts of epochs (started with 100, moved up to 200)
    
Second Attempt: 

![Alt text](image-2.png)

![Alt text](image-3.png)

Third Attempt: 

![Alt text](image-4.png)

![Alt text](image-5.png)
    
4. Summarize the overall results of the deep learning model. Include a recommendation for how a different model could solve this classification problem, and then explain your recommendation.
Overall, the accuracy of my model was only ~72%, which is generally considered to be a good model (<70%). However, I believe better accuracy could be achieved with other models, including random forests as that type of model is less likely to overfit the skew the provided data. 