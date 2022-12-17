import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

def preprocessing(filename = "train.csv"):
    
    print("=================================================")
    print("=========== START PREPROCESSING =================")
    print("=================================================")
    
    
    # Train and test CSV import as Data Frame
    df_train_origin = pd.read_csv(filename)
    if 'Category' in df_train_origin.columns:
        df_train_origin.sort_values(by="Category")
    
    print("========= Loaded data and sorted category=============")
    
    #remove all records that are not located in San Francisco
    df_train_origin = df_train_origin[df_train_origin["Y"] < 38]
    
    print("========= Removed outliers =============")
    
    
    #remove duplicated rows!!!
    df_train_origin = df_train_origin.drop_duplicates()
    
    print("========= Removed duplicated rows =============")
    
    #878049 remaining rows
    
    #Factorize data (assign numbers to different categories)
    #print(df_train_origin.columns)
    #print(df_train_origin["Category"].sort_values(ascending=True))
    if 'Category' in df_train_origin.columns:
        #print('wuhu')
        Y_df = pd.factorize(df_train_origin["Category"])
    else:
        Y_df = [0,0]
        
    print("========= Factorized our Y =============")
    
    #print(Y_df[1])
    #print(Y_df)
    weekdays = pd.factorize(df_train_origin["DayOfWeek"])
    #pdDistricts = pd.factorize(df_train_origin["PdDistrict"])
    
    print("========= Factorized our DayOfWeek =============")
    '''
    addresses = []
    for address in df_train_origin["Address"]:
        if address.lower().find("block") != -1:
            addresses.append(1)
        else:
            addresses.append(0)
    '''
    
    print("========= Start K-Means =============")
    df_coordinates = [df_train_origin["X"], df_train_origin["Y"]]
    df_coordinates = np.array(df_coordinates).T
    
    
    kmeans = KMeans(n_clusters=5, random_state=0).fit(df_coordinates)
    cluster_assignment = kmeans.predict(df_coordinates) #weist jedem Datenpunkt dem entsprechenden Cluster zu (Entweder Cluster 0,1,2,3)
    
    print("========= End K-Means =============")
    
    df_coordinates= pd.DataFrame(df_coordinates)

    centers = pd.DataFrame(kmeans.cluster_centers_[cluster_assignment, :])
    centers.index = df_coordinates.index
    df_coordinates = pd.concat([df_coordinates, centers], axis=1)
    df_coordinates
    
    print("========= Mapped cluster-centers =============")
    
    weekends = (df_train_origin["DayOfWeek"] == "Sunday" ) | (df_train_origin["DayOfWeek"] == "Saturday")
    weekends = weekends.replace(False, 0)
    weekends = weekends.replace(True, 1)
    datetimes =  pd.to_datetime(df_train_origin['Dates'])
    
    print("========= Created feature weekend =============")
    
    data_new = { "DayOfWeek": weekdays[0], "Weekend": weekends,
                 "CoordinateClusters" : cluster_assignment,
                 "X": df_train_origin["X"], "Y": df_train_origin["Y"]}
    
    X_df = pd.DataFrame(data_new)
    L = ['year', 'month', 'quarter', 'hour', 'date', 'minute']
    date_gen = (getattr(datetimes.dt, i).rename(i) for i in L)
    X_df = X_df.join(pd.concat(date_gen, axis=1))
    
    print("========= Generated basic X =============")
    
    #all different police department district
    df_districts = pd.unique(df_train_origin["PdDistrict"])
    
    for district in df_districts:
        autor = district
        district = df_train_origin["PdDistrict"].str.contains(district, case=True)
        #print(day)
        X_df[autor] = district.astype(int)
    
    print("========= One hot Encoded PdDistrict =============")
    
    X_df['ST'] = df_train_origin['Address'].str.contains(" ST", case=True)
    X_df['ST'] = X_df['ST'].replace(False, 0)
    X_df['ST'] = X_df['ST'].replace(True, 1)
    
    X_df['AV'] = df_train_origin['Address'].str.contains(" AV", case=True)
    X_df['AV'] = X_df['AV'].replace(False, 0)
    X_df['AV'] = X_df['AV'].replace(True, 1)
    
    X_df['WY'] = df_train_origin['Address'].str.contains(" WY", case=True)
    X_df['WY'] = X_df['WY'].replace(False, 0)
    X_df['WY'] = X_df['WY'].replace(True, 1)
    
    X_df['TR'] = df_train_origin['Address'].str.contains(" TR", case=True)
    X_df['TR'] = X_df['TR'].replace(False, 0)
    X_df['TR'] = X_df['TR'].replace(True, 1)
    
    X_df['DR'] = df_train_origin['Address'].str.contains(" DR", case=True)
    X_df['DR'] = X_df['DR'].replace(False, 0)
    X_df['DR'] = X_df['DR'].replace(True, 1)
    
    X_df['Block'] = df_train_origin['Address'].str.contains(" Block", case=True)
    X_df['Block']  = X_df['Block'].replace(False, 0)
    X_df['Block']  = X_df['Block'].replace(True, 1)
    
    X_df['crossing'] = df_train_origin['Address'].str.contains(" / ", case=True)
    X_df['crossing']  = X_df['crossing'].replace(False, 0)
    X_df['crossing']  = X_df['crossing'].replace(True, 1)
    
    print("========= Addeds features to X from address =============")
    
    
    df_temp_data_0 = pd.read_csv("2003-2004.csv")[['datetime', 'temp', 'conditions']]
    df_temp_data_1 = pd.read_csv("2005-2006.csv")[['datetime', 'temp', 'conditions']]
    df_temp_data_1['temp'] = (df_temp_data_1['temp'] - 32) * 5/9
    df_temp_data_2 = pd.read_csv("2007-2008.csv")[['datetime', 'temp', 'conditions']]
    df_temp_data_2['temp'] = (df_temp_data_1['temp'] - 32) * 5/9
    df_temp_data_3 = pd.read_csv("2009-2010.csv")[['datetime', 'temp', 'conditions']]
    df_temp_data_4 = pd.read_csv("2011-2012.csv")[['datetime', 'temp', 'conditions']]
    df_temp_data_5 = pd.read_csv("2013-2014.csv")[['datetime', 'temp', 'conditions']]
    df_temp_data_6 = pd.read_csv("2015.csv")[['datetime', 'temp', 'conditions']]
    df_temp_data_0 = df_temp_data_0.append(df_temp_data_1, ignore_index = True).append(df_temp_data_2, ignore_index = True).append(df_temp_data_3, ignore_index = True).append(df_temp_data_4, ignore_index = True).append(df_temp_data_5, ignore_index = True).append(df_temp_data_6, ignore_index = True)
    X_df['date'] = X_df['date'].astype(str)
    df_temp_data_0['datetime'] = df_temp_data_0['datetime'].astype(str)
    X_df = pd.merge(X_df, df_temp_data_0, how='left', left_on='date', right_on='datetime')
    X_df = X_df.drop(['date', 'datetime'], axis=1)
    
    print("========= Imported weather-data =============")
    
    
    df_conditions = pd.unique(X_df["conditions"])
    
    
    for condition in df_conditions:
        condition = str(condition)
        autor = condition
        condition = X_df["conditions"].astype(str).str.contains(condition, case=True)
        #print(day)
        X_df[autor] = condition.astype(int)
    
    
    del X_df["conditions"]
    
    print("========= One hot Encoded conditions =============")
    #with pd.option_context('display.max_columns', None):  # more options can be specified also
       # print(X_df)
    
    X_df["X_centroids"] = df_coordinates.iloc[:,2]
    X_df["Y_centroids"] = df_coordinates.iloc[:,3]
    
   # del X_df["CoordinateClusters"]
    
    #print("========= Added X- and Y- coordinates to X =============")
    
    #print(X_df.isnull().sum())
    
        
    print("=================================================")
    print("=========== END PREPROCESSING =================")
    print("=================================================")
    
    return Y_df, X_df