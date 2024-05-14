## Deep learning model for Alphabet Soup

#### Purpose of the analysis

The analysis aims to develop a binary classifier using machine learning techniques and neural networks to predict the success of applicants funded by Alphabet Soup, a nonprofit foundation. By leveraging features from the provided dataset, the goal is to identify applicants with the highest likelihood of success in their ventures, aiding Alphabet Soup in making informed decisions about funding allocation.

#### Description of the dataset

The dataset comprises over 34,000 organizations that have received funding from Alphabet Soup. It includes various metadata such as identification details, application type, sector affiliation, government classification, use case for funding, organization type, active status, income classification, special considerations, requested funding amount, and a binary indicator of whether the funding was effectively utilized.

#### Preprocess the Data
- Read in the charity_data.csv to a Pandas DataFrame.
- Drop the EIN and NAME columns.
- Determine the number of unique values for each column.
- For columns that have more than 10 unique values, determine the number of data points for each unique value.
- Use the number of data points for each unique value to pick a cutoff point to combine "rare" categorical variables together in a new value, Other, and then check if the replacement was successful.
- Use pd.get_dummies() to encode categorical variables.
- Split the preprocessed data into a features array, X, and a target array, y.
- Use these arrays and the train_test_split function to split the data into training and testing datasets.
- Scale the training and testing features datasets by creating a StandardScaler instance, fitting it to the training data, then using the transform function.

#### Compile, Train, and Evaluate the Model
- Create a neural network model by assigning the number of input features and nodes for each layer using TensorFlow and Keras.
- Create the first hidden layer and choose an appropriate activation function.
- Create an output layer with an appropriate activation function.
- Check the structure of the model.
- Compile and train the model.
- Create a callback that saves the model's weights every five epochs.
- Evaluate the model using the test data to determine the loss and accuracy.
- Save and export your results to an HDF5 file.

#### Optimize the Model
- Create a new Google Colab file and name it AlphabetSoupCharity_Optimization.ipynb.
- Import dependencies and read in the charity_data.csv to a Pandas DataFrame.
- Preprocess the dataset as you did in Step 1. Be sure to adjust for any modifications that came out of optimizing the model.
- Design a neural network model, and be sure to adjust for modifications that will optimize the model to achieve higher than 75% accuracy.
- Save and export your results to an HDF5 file. 
