#even_odd data generator

import pandas as pd
import numpy as np
import random

def generate_df(num_instances=10000, num_features=3, num_dependent_features=2, show_data=False):
    prefix='n_'+str(num_instances)+'_f_'+str(num_features)+'_d_'+str(num_dependent_features)
    output_path = prefix

    #start
    features=[]
    for i in range(num_features):
        f = np.random.randint(0,2,num_instances)
        features.append(f)


    dependent_features_indices = random.sample(range(num_features), num_dependent_features)
    dependent_features=[]
    for index in dependent_features_indices:
        dependent_features.append(features[index])
    dependent_features = np.array(dependent_features).T

    labels=[] #xor for more than 2 features is one if exactly one of the features is 1
    for i in range(num_instances):
        if np.sum(dependent_features[i]) % 2 == 1:
            labels.append(1)
        else:
            labels.append(0)


    #put in dataframe and save
    columns=[]
    df=pd.DataFrame()
    for i in range(num_features):
        if i in dependent_features_indices:
            column = 'D'+str(i)
        else:
            column='F'+str(i)
        df[column] = features[i]
    df['Label'] = labels
    df.to_csv(output_path + '.csv', index=False)
    #df.to_pickle(output_path + '.pkl.gzip')

    if show_data:
        print(dependent_features_indices)
        print(features)
        print(dependent_features)
        print(labels)
        print(df)


    
    
if __name__ == '__main__':
    import sys
    num_instances, num_features, num_dependent_features = sys.argv[1:4]
    generate_df(int(num_instances), int(num_features), int(num_dependent_features))