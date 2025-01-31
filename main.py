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
    townDict = {'Branford': 1, 'Trumbull': 2, 'Madison': 3, 'East Granby': 4, 'Simsbury': 5, 'Bridgewater': 6, 'Beacon Falls': 7, 'Sterling': 8,
                 'Groton': 9, 'Waterford': 10, 'Glastonbury': 11, 'Canton': 12, 'Woodstock': 13, 'Putnam': 14, 'Shelton': 15, 'Ansonia': 16,
                   'Plymouth': 17, 'Avon': 18, 'Canaan': 19, 'Morris': 20, 'Willington': 21, 'North Haven': 22, 'Columbia': 23, 'Thomaston': 24,
                     'Manchester': 25, 'Easton': 26, 'Norfolk': 27, 'Durham': 28, 'Middlebury': 29, 'Chaplin': 30, 'Lisbon': 31, 'East Hampton': 32,
                       'New London': 33, 'Killingly': 34, 'Kent': 35, 'Hampton': 36, 'Rocky Hill': 37, 'Newington': 38, 'Haddam': 39, 'Hartland': 40,
                         'Eastford': 41, 'Mansfield': 42, 'Cornwall': 43, 'Warren': 44, 'Windsor Locks': 45, 'Ledyard': 46, 'Sharon': 47,
                           'New Fairfield': 48, 'Somers': 49, 'Woodbury': 50, 'Burlington': 51, 'Roxbury': 52, 'Torrington': 53, 'Brooklyn': 54,
                             'Cromwell': 55, 'Lebanon': 56, 'Washington': 57, 'Winchester': 58, 'Andover': 59, 'Salem': 60, 'Ashford': 61, 'Bethany': 62,
                               'Middletown': 63, 'Tolland': 64, 'South Windsor': 65, 'Coventry': 66, 'Oxford': 67, 'Norwich': 68, 'Deep River': 69,
                                 'Wilton': 70, 'Colchester': 71, 'Berlin': 72, 'Vernon': 73, 'Canterbury': 74, 'Monroe': 75, 'Stamford': 76,
                                   'Plainfield': 77, 'North Branford': 78, 'Bethel': 79, 'Granby': 80, 'Hamden': 81, 'Newtown': 82, 'Wolcott': 83,
                                     'Stafford': 84, 'Derby': 85, 'Bridgeport': 86, 'Wallingford': 87, 'Westport': 88, 'Portland': 89, 'Killingworth': 90,
                                       'Danbury': 91, 'Cheshire': 92, 'Fairfield': 93, 'Marlborough': 94, 'New Canaan': 95, 'Bozrah': 96,
                                         'West Hartford': 97, 'Darien': 98, 'Old Saybrook': 99, 'Orange': 100, 'East Haven': 101, 'West Haven': 102,
                                           'Preston': 103, 'Westbrook': 104, 'Hartford': 105, 'Meriden': 106, 'Suffield': 107, 'Plainville': 108,
                                             'New Britain': 109, 'Hebron': 110, 'Brookfield': 111, 'East Windsor': 112, 'Lyme': 113, 'New Milford': 114,
                                               'East Hartford': 115, 'Franklin': 116, 'Bethlehem': 117, 'Naugatuck': 118, 'Essex': 119, 'Enfield': 120,
                                                 'Bolton': 121, 'Harwinton': 122, 'Stonington': 123, 'Griswold': 124, 'Watertown': 125, 'Sherman': 126,
                                                   'Bloomfield': 127, 'Chester': 128, 'Old Lyme': 129, 'Southbury': 130, 'Goshen': 131, 'Farmington': 132,
                                                     'Ellington': 133, 'Clinton': 134, 'Waterbury': 135, 'New Hartford': 136, 'Middlefield': 137,
                                                       'Stratford': 138, 'East Lyme': 139, 'Colebrook': 140, 'Litchfield': 141, 'Greenwich': 142,
                                                         'Ridgefield': 143, 'Thompson': 144, 'Guilford': 145, 'Milford': 146, 'Norwalk': 147,
                                                           'New Haven': 148, 'East Haddam': 149, 'Bristol': 150, 'Montville': 151, 'Barkhamsted': 152,
                                                             'Wethersfield': 153, 'Windsor': 154}
    typeDict = {'Single Family': 1, 'Two Family' : 2, 'Three Family' : 3, 'Four Family' : 4, 'Condo' : 5, np.nan: 6}
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
    newData = pd.DataFrame({'Town': [60,60,60,60], 'Type': [1,2,3,5]})
    y_pred = model.predict(newData)


    # Creating the graph
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('Town')
    ax.set_ylabel('Type', labelpad=40)
    ax.set_zlabel('House Price')
    ax.set_zlim(0,5000000)
    ax.set_zticks(np.arange(0,5000000,500000))
    ax.set_yticks([1,2,3,4,5,6,7])
    ax.set_yticklabels(['Single Family', 'Two Family', 'Three Family', 'Four Family', 'Condo', 'Other', ''])
    ax.tick_params(axis='y', rotation=270)

    # Plotting the training data in orange and prediction data in blue
    #ax.scatter(X_test['MedInc'], X_test['AveBedrms'], y_pred, color='blue', label='Actual Data')
    #ax.scatter(X_train['MedInc'], X_train['AveBedrms'], y_train, color='orange', label='Training Data')
    ax.scatter(xAxis['Town'], xAxis['Type'], yAxis, color='red', label='Training Data')
    ax.scatter(newData['Town'], newData['Type'], y_pred, color='blue')

    

    # Getting the range of x values for the line of best fit
    #x1_range = np.linspace(X_test['MedInc'].min(), X_test['MedInc'].max(), 100)
    #x2_range = np.linspace(X_test['AveBedrms'].min(), X_test['AveBedrms'].max(), 100)
    #x1, x2 = np.meshgrid(x1_range, x2_range)

    # Creating a z axis for the line of best fit
    #z = model.predict(np.c_[x1.ravel(), x2.ravel()]).reshape(x1.shape)

    # Draw the line of best fit
    #ax.plot_surface(x1, x2, z, color='orange', alpha=0.5, rstride=100, cstride=100)

    
    ax.set_title('Multiple Linear Regression Best Fit Line (3D)')
    plt.savefig("output.jpg", dpi=300)
    plt.show()

