# importing Flask and other modules
from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing

 
# Flask constructor
app = Flask(__name__)   
 
# A decorator used to tell the application
# which URL is associated function
@app.route('/', methods =["GET", "POST"])
def index():
    if request.method == "POST":
        # getting input with name = fname in HTML form
        first_name = request.form.get("fname")
        # getting input with name = lname in HTML form 
        last_name = request.form.get("lname") 
        return last_name + ", " + first_name

    return render_template("index.html")

def xAxisToInt(xAxis):
    townDict = {i+1: town for i, town in enumerate(set(xAxis['Town']))}
    townDict = {v: k for k, v in townDict.items()}
    typeDict = {i+1: type for i, type in enumerate(set(xAxis['Type']))}
    typeDict = {v: k for k, v in typeDict.items()}
    townList = []
    typeList = []

    for item in xAxis['Town']:
        townList.append(townDict[item])
    
    for item in xAxis['Type']:
        typeList.append(typeDict[item])

    return pd.DataFrame({'Town': townList, 'Type': typeList})

if __name__=='__main__':
    #app.run()
    # Get the data from sklearn and create the model
    california_housing = fetch_california_housing()

    # Create the model
    model = LinearRegression()

    # Get the data from the csv
    rawData = pd.read_csv("ConneticutResidentialSales2001-2022.csv")
    townData = []
    typeData = []
    saleData = []


    # Assign the data (features) to the X and Z axis
    #X = pd.DataFrame(california_housing.data, columns=california_housing.feature_names)
    xAxis = pd.DataFrame({'Town': rawData["Town"], 'Type': rawData["Residential Type"]})
    # Assign the target (home price) to the Y axis
    #y = pd.Series(california_housing.target)
    yAxis = pd.Series(rawData["Sale Amount"])

    print(yAxis)

    # Adding the features for the visual

    ### DECIDE WHICH TWO THINGS TO USE, MAYBE AVG ROOMS AND HOUSE AGE???
    # https://inria.github.io/scikit-learn-mooc/python_scripts/datasets_california_housing.html
    #X = X[['MedInc', 'AveBedrms']]

    # 100% of the data needs to go into training
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1, random_state=1)

    # Actually train the model
    #model.fit(X_train, y_train)
    
    xAxis = xAxisToInt(xAxis)
    xTrain = pd.DataFrame({'Town': list(xAxis['Town'].keys()), 'Type': list(xAxis['Type'].keys())})
    model.fit(xTrain, yAxis)

    # Make a prediction?
    #y_pred = model.predict(X_test)


    # Creating the graph
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Plotting the training data in orange and prediction data in blue
    #ax.scatter(X_test['MedInc'], X_test['AveBedrms'], y_pred, color='blue', label='Actual Data')
    #ax.scatter(X_train['MedInc'], X_train['AveBedrms'], y_train, color='orange', label='Training Data')
    ax.scatter(xAxis['Town'], xAxis['Type'], yAxis, color='red', label='Training Data')

    

    # Getting the range of x values for the line of best fit
    #x1_range = np.linspace(X_test['MedInc'].min(), X_test['MedInc'].max(), 100)
    #x2_range = np.linspace(X_test['AveBedrms'].min(), X_test['AveBedrms'].max(), 100)
    #x1, x2 = np.meshgrid(x1_range, x2_range)

    # Creating a z axis for the line of best fit
    #z = model.predict(np.c_[x1.ravel(), x2.ravel()]).reshape(x1.shape)

    # Draw the line of best fit
    #ax.plot_surface(x1, x2, z, color='orange', alpha=0.5, rstride=100, cstride=100)

    ax.set_xlabel('Town')
    ax.set_ylabel('Type')
    ax.set_zlabel('House Price')
    ax.set_title('Multiple Linear Regression Best Fit Line (3D)')

    plt.show()
