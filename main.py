# importing Flask and other modules
from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Universal town and type dictionaries
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
inputTown = 0
inputType = 0
# Flask constructor
app = Flask(__name__)   
 
# A decorator used to tell the application
# which URL is associated function
@app.route('/', methods =["GET", "POST"])
def index():
    if request.method == "POST":
        # getting input with name = fname in HTML form
        inputTown = int(request.form.get("town"))
        # getting input with name = lname in HTML form 
        inputType = int(request.form.get("type"))
    return render_template("index.html")

def xAxisToInt(xAxis):
    # Create lists for the output
    townList = []
    typeList = []

    # Adding the key for every item into a list sequentially
    for item in xAxis['Town']:
        townList.append(townDict[item])
    
    for item in xAxis['Type']:
        typeList.append(typeDict[item])

    return pd.DataFrame({'Town': townList, 'Type': typeList})

if __name__=='__main__':    
    # Create the model
    model = LinearRegression()

    # Get the data from the csv
    rawData = pd.read_csv("ConneticutResidentialSales2001-2022.csv")
    townData = []
    typeData = []
    saleData = []

    # Assign the data to the X and Z axis and formatting it for training
    xAxis = pd.DataFrame({'Town': rawData["Town"], 'Type': rawData["Residential Type"]})
    xAxis = xAxisToInt(xAxis)
    xTrain = pd.DataFrame({'Town': list(xAxis['Town'].keys()), 'Type': list(xAxis['Type'].keys())})
    
    # Assign the target (home price) to the Y axis
    yAxis = pd.Series(rawData["Sale Amount"])

    # Training the model
    model.fit(xTrain, yAxis)

    # TODO: Get the data from the html form and use it as an input for newData
    # Make a prediction based on the user inputted town and residence type
    newData = pd.DataFrame({'Town': [inputTown], 'Type': [inputType]})
    print(newData)
    y_pred = model.predict(newData)


    # Creating the graph
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('Multiple Linear Regression Best Fit Line (3D)')

    ax.set_xlabel('Town')

    ax.set_ylabel('Type', labelpad=40)
    ax.set_yticks([1,2,3,4,5,6,7])
    ax.set_yticklabels(['Single Family', 'Two Family', 'Three Family', 'Four Family', 'Condo', 'Other', ''])
    ax.tick_params(axis='y', rotation=270)

    ax.set_zlabel('House Price')
    ax.set_zlim(0,5000000)
    ax.set_zticks(np.arange(0,5000000,500000))

    # Plotting the training data in orange and prediction data in blue
    #ax.scatter(xAxis['Town'], xAxis['Type'], yAxis, color='orange', label='Training Data')
    ax.scatter(newData['Town'], newData['Type'], y_pred, color='blue')
    
    # Outputting the graph
    plt.savefig("resources/output.jpg", dpi=300)
    #plt.show()
    app.run()