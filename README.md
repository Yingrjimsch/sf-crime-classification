## San Francisco Crime Classification
 by Noah Lichtencker (lichtnoah), Gabriel Nobel (yingrjimsch), Rebekka von Wartburg (vonwareb)

### Introduction
During twelve weeks, we were shown the basic tools of machine learning in the Machine Learning and
Data Mining module. In numerous practical courses, we were able to get to know the individual
"tools" by getting hands on fictitious examples. Now we get the chance to prove our skills on real data.

### Our Challenge
Our task is to use a dataset of 12 years of crime reports to predict the correct type of crime in San Francisco.
After we received the task, however, the questions then arose very quickly:
* What types of data are we talking about?
* What does the data look like?
* What classifier is the best for this scenario?
* What data missing data would be usefull?


### Data - Analysis

On [Kaggle](https://www.kaggle.com/competitions/sf-crime/ "kaggle") and by looking at the content of the data the following information could be obtained:

* **Category** &#10132; Name (Category) of the incident, our Variable to predict as string representation
* **Dates** &#10132;  Timestamp of the crime incident
* **Descript** &#10132; A description of the incident. (This feature is only available in the training data)
* **DayOfWeek** &#10132; the day of the week as string representation
* **PdDistrict** &#10132; Name of the Police Department District
* **Resolution** &#10132; Resolution of the incident. How was the crime solved (This feature is also only available in the training data)
* **Address** &#10132; Approximate street address of the incident as string representation
* **X** &#10132; Geographical longitude
* **Y** &#10132; Geographical latitude

After knowing what data is available it was to decide if the data is really necessary.
* ✔️ Category --> Key data and our "Y" Value to be found out.
* ✔️ Dates --> can be relevant (e.g. more crime on weekends) //TODO: Plot Crime category per Month
* ❌ Description --> not decisive enough, hard to categorize, not in the test data
* ✔️ DayOfWeek --> can be relevant (e.g. more crime on weekends) //TODO: Plot Crime category per day or per weekend
* ✔️ PdDistrict --> a sort of "clustering" can be very relevant //TODO: Plot Crime category per PdDistrict
* 

### Preprocessing

#### Error- Detection
Now that it was clear what kind of data we were dealing with, we looked for "errors" in the data that could complicate the later model building.
1. Does it have NULL values in the data?
``code for null Detection``
&#10132; obviously it has no NULL values
2. Does it have outliers in the data?
``code for outlier Detection``
 &#10132; Since San Francisco is the latitude 37.7562° and the longitude -122.4430° it could be determined very quickly that there are 67 outliers in the training data.
These are very few wrong values in relation to the whole dataset. Therefore we decided to delete them.
If there had been more data, we could have tried to replace them, for example, with an average value.
3. Does it have duplicate records in the data
``code for outlier Detection``
&#10132; A total of 2323 duplicates were identified in the data. These were also removed from the data set.

#### Data- Preparing
Since we learned from class that a classifier can only deal with numerical values, we now tried to convert the individual features into clever numerical values.
But what are suitable numerical values? Can features be split up to extract new features?
Should individual features be combined to obtain the greatest benefit?
Or could even additional features (e.g. from the web) be added?

* Entfernen von Description und Resolution, da nur in den Trainingsdaten vorhanden.
Nutzlos um gutes Testergebnis zu erhalten
* Visualisierung der Daten -> Einfügen einiger Plots zur Visualisierung
Welche Features zeigen Relevanz; Unausgeglichenheit der Y-daten -> balancieren
* Clustering der X-Y Daten (KMeans)
* Aussplitten der Dates in year, month, quarter, hour, minute
* Zusätzliches binäres Feature ob Wochentag oder am Wochenende
* PdDistricts werden in binäre Spalten ufgeteilt
* Addressen werden binär aufgeteilt, ob Block oder Strasse
* Hinzunehmen von wetterdaten: temp und?
* Final resultieren 22 Features um zu trainieren
* 


### Model Selection
* **Classifier 1**
  * RandomForest
    * Characteristics: To test the RandomForest for our data we chose a max depth of 27 (for each feature one depth deeper) and a total of 150 trees to go over the data.
    * Why: Random Forest seemed a good fit to predict and evaluate our data and can have similar results to a Neural Network
* **Classifier 2**
  * Support Vector Machine
    * Characteristics: 
    * Why: 
* **Classifier 3**
  * Neural Network
    * Characteristics:
    * Why:
  
### Training the Model

* **Classifier 1**
  * 27 depth seemed inefficient. so we tried decreasing the depth to see which depth seemed best. The score seemed best with a depth of 21
* **Classifier 2**
  * The Support Vector Machine turned out to be a failure. It processed and evaluated data for two days, in the end we canceled the process. We tried to reduce the data to 5000. 
* **Classifier 3**
  * 

### Model Evaluation

* **Classifier 1**
  * In the end the random forest was inferior to our NN solution. We could have probably gone to a similiar score with more estimator trees, but that would have taken a longer processing time that a Neural Network could achieve.
* **Classifier 2**
  * The Support Vector Machines final loss score was somewhere between 5.0-5.6. In the end we decided to give up on the Model.
* **Classifier 3**
  *
