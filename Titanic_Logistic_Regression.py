import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the passenger data
passengers = pd.read_csv("passengers.csv")

# Inspection
#print(passengers.head())
#print("This is the column names of the dataframe:")
#print(passengers.columns)
#print("This is the info of the data in the dataframe")
#print(passengers.info())
#print("This is the description of the data present:")
#print(passengers.describe())
#print("This is the data type of each column")
#print(passengers.dtypes)
# Update sex column to numerical

#Given the saying, “women and children first,” Sex and Age seem like good features to predict survival. Let’s map the text values in the Sex column to a numerical value. Update Sex such that all values female are replaced with 1 and all values male are replaced with 0.
passengers['Sex'] = passengers['Sex'].map({'female' : 1, 'male': 0})
#print(passengers['Sex'].head())

# Looking at the Age column
#print(passengers['Age'])

# Fill the nan values in the age column
mean_age = passengers['Age'].mean().round(0)
#print(mean_age)
passengers['Age'].fillna(value = mean_age, inplace=True)
#print(passengers['Age'])

#Given the strict class system onboard the Titanic, let’s utilize the Pclass column, or the passenger class, as another feature. Create a new column named FirstClass that stores 1 for all passengers in first class and 0 for all other passenger
# Create a first class column
passengers['FirstClass'] = passengers['Pclass'].apply(lambda x: 1 if x == 1 else 0)
#print(passengers['FirstClass'])

# Create a second class column
passengers['SecondClass'] = passengers['Pclass'].apply(lambda x: 1 if x == 2 else 0)
#print(passengers['SecondClass'])

# Inspecting the DF to see if all the changes have been made
#print(passengers.columns)

# Select the desired features
features = passengers[['Sex', 'Age', 'FirstClass', 'SecondClass']]
#Inspecting the features dataframe
#print(features)

# Creating a Survived array
survival = passengers['Survived']
# Inspecting teh survival df
#print(survival)

# Perform train, test, split
feature_train, feature_test, validation_train, validaton_test = train_test_split(features, survival, test_size = 0.2)
#We’ll use the training set to train the model and the test set to evaluate the model.

# Scale the feature data so it has mean = 0 and standard deviation = 1
# we are using Regularization to scale the data rather than normalization
scale = StandardScaler()
#.fit_transform() it on the training features
feature_train = scale.fit_transform(feature_train)
# use .transform on the test features
feature_test = scale.transform(feature_test)

#inspecting the data
#print(feature_train)
#print(feature_test)

# Create and train the model
model = LogisticRegression()
# Training the model
model.fit(feature_train, validation_train)

# getting the coeffiencts and the intercept from the model
#print(model.intercept_)
#print(model.coef_)

# Score the model on the train data
#Scoring the model on the training data will run the data through the model and make final classifications on survival for each passenger in the training set. The score returned is the percentage of correct classifications, or the accuracy.
print(model.score(feature_train, validation_train))
# returns 0.77 

# Score the model on the test data
# Similarly, scoring the model on the testing data will run the data through the model and make final classifications on survival for each passenger in the test set.
print(model.score(feature_test, validaton_test))
# returns 0.810

# Analyze the coefficients
#print("This is the coefficients determined by the model: " + str(model.coef_))
#print("The respective features are as follows: ")
print(list(zip(['Sex','Age','FirstClass','SecondClass'],model.coef_[0])))

# - Features with larger, positive coefficients will increase the probability of a data sample belonging to the positive class
# - Features with larger, negative coefficients will decrease the probability of a data sample belonging to the positive class
# - Features with small, positive or negative coefficients have minimal impact on the probability of a data sample belonging to the positive class

# As we can see, the sex of the passengers has the highest probability that a passengers survival depended on it

# Sample passenger features
Jack = np.array([0.0,20.0,0.0,0.0])
Rose = np.array([1.0,17.0,1.0,0.0])
You = np.array([1.0,25.0,0.0,0.0])

# Combine passenger arrays
sample_passengers = np.array([Jack, Rose, You])

# Scale the sample passenger features
# Since our Logistic Regression model was trained on scaled feature data, we must also scale the feature data we are making predictions on.
sample_passengers = scale.transform(sample_passengers)
# Investigating the sample passengers after it has been scaled
#print(sample_passengers)
# Make survival predictions!
print(model.predict(sample_passengers))

# The probabilities that let to these predictions
print(model.predict_proba(sample_passengers))

# The 1st column is the probability of a passenger perishing on the Titanic
# The 2nd column is the probability of a passenger surviving the sinking (which was calculated by our model to make the final classification decision).
