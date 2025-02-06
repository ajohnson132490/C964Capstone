# importing Flask and other modules
import matplotlib
matplotlib.use('Agg')
from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Global dictionaries
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

# Globar variables 
model = LinearRegression()
ttData = pd.DataFrame()
ttTrain = pd.DataFrame()
sales = pd.DataFrame()

# Flask constructor
app = Flask(__name__)   
 
# A decorator used to tell the application
# which URL is associated function
@app.route('/', methods =["GET", "POST"])
def index():
    value = 0
    title = ""
    if request.method == "POST":
        # getting input with name = fname in HTML form
        inputTown = int(request.form.get("town"))
        # getting input with name = lname in HTML form 
        inputType = int(request.form.get("type"))
        value = float(scatterPlot(inputTown, inputType)[0])
        pieChart(inputTown)
        title = histogram(inputTown, inputType)
    return render_template("index.html", val="$" + str(value), header=title)

def ttDataToNums(ttData):
    # Create lists for the output
    townList = []
    typeList = []

    # Adding the key for every item into a list sequentially
    for item in ttData['Town']:
        townList.append(townDict[item])
    
    for item in ttData['Type']:
        typeList.append(typeDict[item])

    return pd.DataFrame({'Town': townList, 'Type': typeList})

def scatterPlot(inputTown, inputType):
    # Make a prediction based on the user inputted town and residence type
    newData = pd.DataFrame({'Town': [inputTown], 'Type': [inputType]})
    y_pred = model.predict(newData)

    # Creating the graph
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('Multiple Linear Regression')

    ax.set_xlabel('Town')
    ax.set_xlim([1.0, 160.0])

    ax.set_ylabel('Type', labelpad=40)
    ax.set_yticks([1,2,3,4,5,6,7])
    ax.set_ylim([1.0, 7.0])
    ax.set_yticklabels(['Single Family', 'Two Family', 'Three Family', 'Four Family', 'Condo', 'Other', ''])
    ax.tick_params(axis='y', rotation=270)

    ax.set_zlabel('House Price')
    ax.set_zlim(0,5500000)
    ax.set_zticks(np.arange(0,5500000,500000))

    # Plotting the prediction data
    ax.scatter(newData['Town'], newData['Type'], y_pred, color='blue')
    plt.savefig("static/prediction.jpg", dpi=300)

    # Plotting the training data
    ax.scatter(ttData['Town'], ttData['Type'], sales, color='orange')

    # Outputting the graph
    plt.savefig("static/scatter.jpg", dpi=300)

    return y_pred
  
def pieChart(inputTown):
    # Creating the graph
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111,)
    ax.set_title('Most Frequently Sold Home Type in Submitted Town')

    # Getting the data for the town submitted
    homesInTown = []
    for index, row in ttData.iterrows():
        if (row['Town'] == inputTown):
            homesInTown.append(row['Type'])
    
    # Formatting and Plotting the data 
    labelNums, data = np.unique(homesInTown, return_counts=True)
    label=[]
    for i in labelNums:
        if i == 1:
          label.append('Single Family')
        elif i == 2:
          label.append('Two Family')
        elif i == 3:
          label.append('Three Family')
        elif i == 4:
          label.append('Four Family')
        elif i == 5:
          label.append('Condo')
        elif i == 6:
          label.append('Other')

    ax.pie(data, labels=label)
    
    # Outputting the graph
    plt.savefig("static/pie.jpg", dpi=300)

def histogram(inputTown, inputType):
    # SHOWS THE NUMBER OF TIMES A HOME OF THE SELECTED TYPE HAS BEEN SOLD IN THIS TOWN
    # Creating the graph
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111)

    # Creating an inverse dictionary for reverse lookup
    tempTownDict = {v: k for k, v in townDict.items()}
    tempTypeDict = {v: k for k, v in typeDict.items()}
    if (inputType == 6):
      ax.set_title("Other Homes Sold In " + str(tempTownDict[inputTown])) 
    else:
      ax.set_title(str(tempTypeDict[inputType]) + " Homes Sold In " + str(tempTownDict[inputTown]))

    homesInTown = []
    for index, row in ttData.iterrows():
        if (row['Type'] == inputType):
            homesInTown.append(row['Town'])

    n, bins, bars = ax.hist(homesInTown, 154, color='orange')
    bars[inputTown-1].set_facecolor('blue')
    
    # Outputting the graph
    plt.savefig("static/hist.jpg", dpi=300)
    
    return ax.get_title()
if __name__=='__main__':
  # Get the data from the csv
  rawData = pd.read_csv("ConneticutResidentialSales2001-2022.csv")

  # Assign the data to the X and Z axis and formatting it for training
  ttData = ttDataToNums(pd.DataFrame({'Town': rawData["Town"], 'Type': rawData["Residential Type"]}))
  ttTrain = pd.DataFrame({'Town': list(ttData['Town'].keys()), 'Type': list(ttData['Type'].keys())})
  
  # Assign the target (home price) to the Y axis
  sales = pd.Series(rawData["Sale Amount"])

  # Training the model
  model.fit(ttData, sales)    
  app.run()