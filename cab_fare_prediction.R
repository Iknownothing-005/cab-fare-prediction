# setting working directory 
setwd("C:/Users/Neeraj/Desktop/project/project2")

#importing libraries 
x = c("VIM","lubridate","corrgram","C50","Information","e1071","ROSE","caret","dummies","sequences","randomForest","MASS","ggplot2","gbm","DMwR","unbalanced","rpart","sampling")
lapply(x,require,character.only = T)
library(dplyr)

#importing data
data = read.csv("train_cab.csv",header = T, na.strings = c(""," ","NA"))

#Exploratory data analysis 
#plotting box-plot to check outliers 

pickup_longitude = data$pickup_longitude
dropoff_longitude = data$dropoff_longitude


boxplot(pickup_longitude, dropoff_longitude ,
        main = "lonitude_outlier Analysis",
        at = c(1,2),
        names = c("pickup_longitude", "dropoff_longitude"),
        las = 2,
        col = c("orange","red"),
        border = "brown",
        horizontal = TRUE,
        notch = TRUE
)

pickup_latitude = data$pickup_latitude
dropoff_latitude = data$dropoff_latitude

boxplot(pickup_latitude, dropoff_latitude ,
        main = "latitude_outlier Analysis",
        at = c(1,2),
        names = c("pickup_latitude", "dropoff_latitude"),
        las = 2,
        col = c("orange","red"),
        border = "brown",
        horizontal = TRUE,
        notch = TRUE
)





passenger_count = data$passenger_count

boxplot(passenger_count,
        main = "passenger_outlier Analysis",
        names = c("passenger_count"),
        las = 2,
        col = c("orange","red"),
        border = "brown",
        horizontal = TRUE,
        notch = FALSE
)

glimpse(data)

# since Signed Degree Format latitude ranges from -90 to 90 there by removing outlier 
data  = data[!rowSums(data[4] > 90,na.rm = FALSE, dims = 1L),]
data = data[!rowSums(data[4] < - 90,na.rm = FALSE, dims = 1L),]
data = data[!rowSums(data[6] > 90,na.rm = FALSE, dims = 1L),]
data = data[!rowSums(data[6] < -90,na.rm = FALSE, dims = 1L),]

# since Signed Degree Format longitude ranges from -180 to 180 thererby removing outliers
data = data[!rowSums(data[3] > 180,na.rm = FALSE, dims = 1L),]
data = data[!rowSums(data[3] < - 180,na.rm = FALSE, dims = 1L),]
data = data[!rowSums(data[5] > 180,na.rm = FALSE, dims = 1L),]
data = data[!rowSums(data[5] < -180,na.rm = FALSE, dims = 1L),]


#Since maximum passenger in a cab can be 7 therefore removing all the values greater than that 
data  = data[!rowSums(data[7] > 7,na.rm = TRUE, dims = 1L),]

# dropping the fare amount above 400 dollars
data$fare_amount = as.numeric(as.character(data$fare_amount, na.rm = TRUE))
data  = data[!rowSums(data[1] > 400,na.rm = TRUE, dims = 1L),] 

#creating a distance column from latitude and longitude cordinates 
p = 0.017453292519943295    #Math.PI / 180
data$distance = 0.5 - cos((data$dropoff_latitude - data$pickup_latitude) * p)/2 + 
  cos(data$pickup_latitude * p) * cos(data$dropoff_latitude * p) * 
  (1 - cos((data$dropoff_longitude - data$pickup_longitude) * p))/2

data$distance = 12742 * asin(sqrt(data$distance))

#removing the string UTC from datetime column
data$pickup_datetime = sub("UTC", "", (data$pickup_datetime))


#coverting datetime column to POSIXt format
data$pickup_datetime = strptime(as.character(data$pickup_datetime),"%Y-%m-%d %H:%M:%S")

glimpse(data)

#converting all zeros to na
data[c(1,3,4,5,6,7,8)][data[c(1,3,4,5,6,7,8)] == 0] = NA

#creating missing value data containing missing value counts for each columns 
missing_val = data.frame(apply(data,2,function(x){sum(is.na(x))}))

missing_val$columns = row.names(missing_val)
row.names(missing_val) = NULL
names(missing_val)[1] = "Missing_value_count"

missing_val = missing_val[,c(2,1)]

#extracting dates,weekdays,years and months from datetime column

data$weekday = weekdays.POSIXt(data$pickup_datetime)
data$year = year(data$pickup_datetime)
data$hour = hour(data$pickup_datetime)
data$day = day(data$pickup_datetime)
data$month = months.POSIXt(data$pickup_datetime)


#dropping redundant variable from data 
data = within(data, rm(pickup_datetime))

#creating missing value data containing missing value counts for each columns 
missing_val = data.frame(apply(data,2,function(x){sum(is.na(x))}))

missing_val$columns = row.names(missing_val)
row.names(missing_val) = NULL
names(missing_val)[1] = "Missing_value_count"

missing_val = missing_val[,c(2,1)]

#removing columns containing only one na
data = data[!is.na(data$month), ]
data = data[!is.na(data$day), ]
data = data[!is.na(data$hour), ]
data = data[!is.na(data$weekday), ]
data = data[!is.na(data$year), ]



#creating missing value data containing missing value counts for each columns 
missing_val = data.frame(apply(data,2,function(x){sum(is.na(x))}))

missing_val$columns = row.names(missing_val)
row.names(missing_val) = NULL
names(missing_val)[1] = "Missing_value_count"

#checking data info
glimpse(data)

#Imputing Missing Value 
#since distribution of all variables not normal knn-imputation is used to impute missingvalues 
data$fare_amount[is.na(data$fare_amount)] = median(data$fare_amount, na.rm = T)
data$pickup_longitude[is.na(data$pickup_longitude)] = median(data$pickup_longitude, na.rm = T)
data$pickup_latitude[is.na(data$pickup_latitude)] = median(data$pickup_latitude, na.rm = T)
data$dropoff_longitude[is.na(data$dropoff_longitude)] = median(data$dropoff_longitude, na.rm = T)
data$dropoff_latitude[is.na(data$dropoff_latitude)] = median(data$dropoff_latitude, na.rm = T)
data$passenger_count[is.na(data$passenger_count)] = median(data$passenger_count, na.rm = T)
data$distance[is.na(data$distance)] = median(data$distance, na.rm = T)

#rounding of passenger counts to zero
data$passenger_count = round(data$passenger_count,digits=0)

#creating missing value data containing missing value counts for each columns 
missing_val = data.frame(apply(data,2,function(x){sum(is.na(x))}))

missing_val$columns = row.names(missing_val)
row.names(missing_val) = NULL
names(missing_val)[1] = "Missing_value_count"

#coverting segregrating hours to group and creating dummies for data 
data$morning = ifelse(data$hour >= 6 & data$hour < 12,1,0)
data$evening = ifelse(data$hour >= 17 & data$hour < 20,1,0)
data$night = ifelse(data$hour >= 20 & data$hour < 23,1,0)
data$lateNight = ifelse(data$hour >= 0 & data$hour < 3,1,0)
data$earlyMorning = ifelse(data$hour >= 3 & data$hour < 6,1,0)

#dropping hour column
data = within(data, rm(hour))

#creating dummy variables 
glimpse(data)

data$month = as.factor(data$month)
data$weekday = as.factor(data$weekday)
data$year = as.factor(data$year)


data = createDummyFeatures(data, cols = "month" )
data = createDummyFeatures(data, cols = "weekday" )
data = createDummyFeatures(data, cols = "year" )

#dropping one dummy of each category to prevent dummy variable trap
data = within(data, rm(earlyMorning))
data = within(data, rm(month.August))
data = within(data, rm(year.2015))
data = within(data, rm(weekday.Monday))

#normalising data frame
data[,2:8] = (data[,2:8]-min(data[,2:8]))/(max(data[,2:8])-min(data[,2:8]))

#feature selection 
#correlation plot
corrgram(data[,2:8], order = F,
         upper.panel = panel.pie, text.panel = panel.txt, main = "Correlation plot")


##chi_square test
for (i in 9:35) 
{
  print(names(data)[i])
  print(chisq.test(table(data$fare_amount,data[,i])))
}

data_reduced  = subset(data,select = -c(2,3,4,5,9,10,11,13,14,15,17,18,21,25,26,27,28,29))

#creating train and test split
library(caTools)
set.seed(123)

#using multiple linear regression of data 
multi_regressor = lm(fare_amount~., data = data)
step(multi_regressor, direction = "backward", trace= TRUE ) 

#creating new data based on above reslult
data_lm = data[,c("fare_amount","pickup_longitude","pickup_latitude",
                  "dropoff_longitude","distance","morning","evening",
                  "night","month.February","month.January","month.July",
                  "month.June","month.March","month.October","weekday.Saturday",
                  "year.2009","year.2010","year.2011","year.2012","year.2013")]

#creating train test split for data_lm
split =  sample.split(data_lm$fare_amount, SplitRatio = 0.65)
train_set = subset(data_lm, split == TRUE)
test_set = subset(data_lm,split == FALSE)

multi_regressor = lm(fare_amount~., data = train_set)

y_pred = predict(multi_regressor, test_set[,2:20])

#checking RMSE and rsquared 

measureRMSE(test_set$fare_amount,y_pred)
measureRSQ(test_set$fare_amount,y_pred)


#creating train test split for data_lm
split =  sample.split(data_reduced$fare_amount, SplitRatio = 0.65)
train_set1 = subset(data_reduced, split == TRUE)
test_set1 = subset(data_reduced,split == FALSE)



#using svr model for prediction
regressor_svr = svm(formula = fare_amount~.,
                    data = train_set1,
                    type = "eps-regression")


y_pred1 = predict(regressor_svr, test_set1[,2:17])

RMSE(test_set1$fare_amount,y_pred1)

measureRSQ(test_set1$fare_amount,y_pred1)
measureRMSE(test_set1$fare_amount,y_pred1)

#using random forest for on our data
rf_regressor  = randomForest(x = train_set1[,2:17],
                             y = train_set1$fare_amount,
                             ntree = 100)

y_pred2 = predict(rf_regressor, test_set1[,2:17])

measureRSQ(test_set1$fare_amount,y_pred2)
measureRMSE(test_set1$fare_amount,y_pred2)
