import pandas as pd
import numpy as np
from sklearn.cluster import KMeans


def preprocessing():
    # Train and test CSV import as Data Frame
    df_train_origin = pd.read_csv("train.csv")
    
    #remove all records that are not located in San Francisco
    df_train_origin = df_train_origin[df_train_origin["Y"] < 38]
    
    #remove duplicated rows!!!
    df_train_origin = df_train_origin.drop_duplicates()
    
    #878049 remaining rows
    
    #Factorize data (assign numbers to different categories)
    Y_df = pd.factorize(df_train_origin["Category"])
    weekdays = pd.factorize(df_train_origin["DayOfWeek"])
    #pdDistricts = pd.factorize(df_train_origin["PdDistrict"])
    addresses = []
    for address in df_train_origin["Address"]:
        if address.lower().find("block") != -1:
            addresses.append(1)
        else:
            addresses.append(0)
            
    df_coordinates = [df_train_origin["X"], df_train_origin["Y"]]
    df_coordinates = np.array(df_coordinates).T
    
    
    kmeans = KMeans(n_clusters=90, random_state=0).fit(df_coordinates)
    cluster_assignment = kmeans.predict(df_coordinates) #weist jedem Datenpunkt dem entsprechenden Cluster zu (Entweder Cluster 0,1,2,3)
    
            
    weekends = (df_train_origin["DayOfWeek"] == "Sunday" ) | (df_train_origin["DayOfWeek"] == "Saturday")
    weekends = weekends.replace(False, 0)
    weekends = weekends.replace(True, 1)
    datetimes =  pd.to_datetime(df_train_origin['Dates'])
    
    data_new = { "DayOfWeek": weekdays[0], "Weekend": weekends,
                 "Address": addresses, "CoordinateClusters" : cluster_assignment}
    X_df = pd.DataFrame(data_new)
    L = ['year', 'month', 'quarter', 'hour', 'date', 'minute']
    date_gen = (getattr(datetimes.dt, i).rename(i) for i in L)
    X_df = X_df.join(pd.concat(date_gen, axis=1))
    
    #all different police department district
    df_districts = pd.unique(df_train_origin["PdDistrict"])
    
    for district in df_districts:
        autor = district
        district = df_train_origin["PdDistrict"].str.contains(district, case=True)
        #print(day)
        X_df[autor] = district.astype(int)
    
    
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
    
    
    df_conditions = pd.unique(X_df["conditions"])
    
    
    for condition in df_conditions:
        condition = str(condition)
        autor = condition
        condition = X_df["conditions"].astype(str).str.contains(condition, case=True)
        #print(day)
        X_df[autor] = condition.astype(int)
    
    
    del X_df["conditions"]
    
    #with pd.option_context('display.max_columns', None):  # more options can be specified also
       # print(X_df)
    
    
    #print(X_df.isnull().sum())
    return Y_df[0], X_df
