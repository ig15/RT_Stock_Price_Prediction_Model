

#libraries used by me
import threading
import pandas as pd
import pandas_datareader.data as web
import datetime
from sklearn import linear_model
from time import sleep
import joblib

import yfinance as yf

data = yf.download("ES=F", start="2022-01-01", end="2023-01-01")


# this class is to give colors to the print statements/using ASCII codes of the colors
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'



# This function will give the price after the delay of sleep_time units of seconds
def getprice(label="ES=F", sleep_time=1):
    # Pause for sleep_time number of seconds #sleeptime=30 will predict values for the next 30 seconds and the next 30 seconds
    sleep(sleep_time)
    # Using yahoo finance to get the stock market live data
    data3 = pd.DataFrame(index=['A'], columns=['x'])
    quote = web.get_quote_yahoo(label) #a dataframe that will be given as an output from this get code function
    price = quote["price"].values[0]
    data3.xs('A')['x'] = price
    data3.to_csv('actual_price.csv', mode='a', index=False, header=False)
    current_time = datetime.datetime.now() #a function to know at what clock time we are acessing the stock price
    return tuple([price, current_time])


# This function trains the model using the input data(dataframe)
def train(input): #this train function will basically use that linear regression model from the SQL library
    print("\nModel updating...", end=" ")
    # We take last column of the features as target and rest are taken as attributes
    featureMat = input.iloc[:, : len(input.columns) - 1]
    label = input[input.columns[-1]]
    # Here we are using linear regression model
    model = linear_model.LinearRegression()
    model.fit(featureMat, label)
    # Model is being written/saved on the hard drive for later use
    joblib.dump(model, "modelLR.pkl")
    print("[Completed]")


                        #  Training over  #
                       #  we are making here a data frame in real time and we are passing that data frame to our train function 

# Increase the values of these variables to get improved results
# but if you will increase them then you have to wait for
#
#               (number_of_features X training_record_criteran) X steep_time units seconds
#
# for first training

number_of_features = 5  # This indicates how many columns the dataframe will have.
training_record_criterian = 5  # This decides how frequently the model will be retrained/updated 
#[Here: 5 new features/records will enter our dataframe -> model will be retrained]
#THUS AFTER FIRST TIME TRAINING, THE MODEL IS UPDATED AFTER A SERIES OF 25 PREDICTIONS THAT IS 25 PREDICTION LINES!(Time dekhna pdega)

number_of_predictions = 3 # Tells how many prediction in series you want/price of stock in the next 3 seconds



data = pd.DataFrame(columns=range(number_of_features))  # creating an empty dataframe

predict_input = list()


while 1:

    feature = list()  # stores the features for a single record for dataframe
    
    for i in range(number_of_features):

        price = getprice()[0] #getprice was defined above
        feature.append(price)
        predict_input.append(price)
        data2 = pd.DataFrame(index=['A'], columns=['x'])

        try:  # this will throw exception in two cases:
            # 1> model is not yet trained and saved
            # 2> model prediction is not working. 
            #In these cases it will say please wait while the model is getting ready

            first_predict = True  # flag for detecting the first prediction in predicted series
            model = joblib.load("modelLR.pkl")  # trying to open the saved model (can throw exception)/if the model is not present in the hard disk it will jump to except
            print("")
            inputlist = predict_input.copy()  # copying the list to make the prediction if model is ready
            #   printing latest 3 prices
            for feature_value in inputlist[-(3):]:
                print(f" --> ", int(feature_value * 100) / 100, end=" ")
            #   taking the latest price
            price = getprice(sleep_time=0)[0]
            #   Starting the predictions
            for i in range(number_of_predictions):
                pre_price = model.predict([inputlist[-(number_of_features - 1):]])
                #   printing the predicted values one by one in the series
                print(f"{bcolors.OKBLUE} --> ", int(pre_price[0] * 100) / 100, end=" ")
                #   This block will only run for the first prediction in series
                if first_predict:
                    # When prediction tells about increase in price
                    if pre_price[0] - inputlist[-1] > 0:
                        print(f"{bcolors.OKGREEN}  \u2191", end="")
                        #   Calculating the % of increase the program predicts and printing.
                        print(f"[", int((pre_price[0] - price) * 1000000 / price) / 10000, "%] ", end=" ")
                        print(f"    Real: ", price, end="")
                    # When prediction says that no change will happen
                    elif pre_price[0] - inputlist[-1] == 0:
                        print(f" \u2022", end="")
                        print(f"[", int((pre_price[0] - price) * 1000000 / price) / 10000, "%] ", end=" ")
                        print(f" Real: ", price, end="")
                    # When prediction is about decrease in price
                    else:
                        print(f"{bcolors.FAIL}   \u2193", end="")
                        print(f"[", int(-(pre_price[0] - price) * 1000000 / price) / 10000, "%] ",
                              end=" ")
                        print(f" Real: ", price, end="")
                    # Next statement talk about what happened actually
                    if price - inputlist[-1] > 0:
                        print(f"  \u2191", end=" ")
                    elif price - inputlist[-1] == 0:
                        print(f" \u2022", end="")
                    else:
                        print(f"  \u2193", end=" ")

                    first_predict = False


                #   pushing the predicted price in the back of the input array..
                #   it will be used in predicting next element in the series
                inputlist.append(pre_price[0])
                data2.xs('A')['x'] = pre_price[0]
                data2.to_csv('predicted_price.csv', mode='a', index=False, header=False)
        except:

            print("Please Wait while the model is getting ready...")
    #   Adding the feature in the dataframe
    
    data.loc[len(data.index)] = feature
    #data.to_csv('GFG.csv', mode='a', index=False, header=False)
        #   If number of elements present in the dataframe is multiple of training record criterian, the retraining
    if len(data.index) % training_record_criterian == 0:
        # print(data)
        #   training in separate thread
        trainer = threading.Thread(target=train, args=(data,))
        trainer.start()
        trainer.join()
        
        