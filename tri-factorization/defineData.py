import pandas as pd
import numpy as np
from numpy import random as rand


def getRandomSample(posible_candidates, column_target, n_candidates):
    posible_candidates = posible_candidates.drop_duplicates(subset = [ column_target ], keep = 'first' ) 
    posible_candidates = posible_candidates[ column_target ]
    max_posible_candidates = len( posible_candidates.index )
    l_candidates = list( posible_candidates )
    if n_candidates < max_posible_candidates:
        candidate_index = np.arange( max_posible_candidates )
        rand.shuffle( candidate_index )
        moviesAux = candidate_index[:n_candidates ]
        l_candidates = [ l_candidates[ i ] for i in moviesAux ]
    else:
        n_candidates = max_posible_candidates
    return l_candidates, n_candidates



def getRandomSampleByMovies(data, n_users, n_movies):
    #ordeno las movies por mayor cantidad de ratings
    groupby_movies = data.groupby( data[ data.columns[ 1 ] ] ).size().to_frame(name='count').reset_index()
    groupby_movies = groupby_movies.sort_values(['count'], ascending=False )
    groupby_movies = groupby_movies.iloc[0:n_movies]
    
    print("Group by movies...{}".format(len(groupby_movies.index)))
    print( groupby_movies.head() )
    
    l_movies = list( groupby_movies[ groupby_movies.columns[ 0 ] ] )
    
    #seleccione aleatoriamente n usuarios
    posible_candidates = data.loc[ data[ data.columns[ 1 ] ].isin( l_movies ) ]
    groupby_users = posible_candidates.groupby( posible_candidates[ posible_candidates.columns[ 0 ] ] ).size().to_frame(name='count').reset_index()
    groupby_users = groupby_users.loc[ groupby_users[ groupby_users.columns[ 1 ] ] >= 40 ]
    
    print("Group by users...{}".format( len( groupby_users.index) ) )
    print( groupby_users.head() )
    l_users, n_users = getRandomSample( groupby_users, groupby_users.columns[ 0 ], n_users )
       
    return l_users, l_movies


def getRandomSampleByUsers(data, n_users, n_movies):
     
    #ordeno los usuarios con mayor cantidad de ratings
    groupby_users = data.groupby( data[ data.columns[ 0 ] ] ).size().to_frame(name='count').reset_index()
    groupby_users = groupby_users.loc[ groupby_users[ groupby_users.columns[ 1 ] ] >= 40 ]
        
    groupby_users = groupby_users.sort_values(['count'], ascending=False)
    print("Group by users...")
    print( groupby_users.head() )
    
    l_users = list( groupby_users[ groupby_users.columns[ 0 ] ] )
    max_posible_users = len( groupby_users.index )
    if n_users < max_posible_users:
        users_index = np.arange( len( groupby_users.index ) )
        rand.shuffle(users_index)
        usersAux = users_index[ :n_users ]
        l_users = [ l_users[ i ] for i in usersAux ]
    else:
        n_users = max_posible_users

    #selecciono n movies aleatoriamente
    posible_candidates = data.loc[ data[ data.columns[ 0 ] ].isin( l_users ) ]
    l_movies, n_movies = getRandomSample( posible_candidates, data.columns[ 1 ], n_movies )
    

    return l_users, l_movies



def getSample(data, l_users, l_movies):
    
    dataUsers = data[ data[ data.columns[ 0 ] ].isin( l_users ) ]
    dataMovies = data[ data[ data.columns[ 1 ] ].isin( l_movies ) ]
    dataSample = pd.merge( dataUsers, dataMovies, how='inner', left_on=list(data.columns[ 0:2 ]), right_on=list(data.columns[ 0:2 ]) )
    dataSample = dataSample[ list(dataSample.columns[0:4] ) ]
    dataSample = dataSample.rename(columns={dataSample.columns[0]:data.columns[0], dataSample.columns[1]:data.columns[1], dataSample.columns[2]:data.columns[2], dataSample.columns[3]:data.columns[3]  })
    print("dataSample...")
    print( dataSample.head() )
    return dataSample  


def getTestSet( data, l_users, len_dtest ):
    
    posible_userTest = np.arange( len( l_users ) )
    rand.shuffle(posible_userTest)
    users_index = posible_userTest[:len_dtest]
    users_test = [ l_users[i] for i in users_index ]
    users_training = []
    for i in l_users:
        if not(i in users_test):
            users_training.append(i)
    Dtraining = data.loc[ data[ data.columns[ 0 ] ].isin( users_training ) ]
    Dtest = data.loc[ data[ data.columns[ 0 ] ].isin( users_test ) ]
    return Dtraining, Dtest, users_training, users_test
    

def hideRatings(Dtest):
    
    x, y = Dtest.shape
    Dtest_hide = np.zeros(Dtest.shape)
    
    for i in range(x):
        ratings_index = np.where(Dtest[i]>0)
        posible_no_hide = np.arange( len( ratings_index[0] ) )
        rand.shuffle( posible_no_hide )
        no_hide_index = posible_no_hide[:15]
        rating_no_hide = [ ratings_index[0][k] for k in no_hide_index ]
        for j in rating_no_hide:
            Dtest_hide[ i ][ j ] = Dtest[ i ][ j ]
    
    return Dtest_hide    
        
                
def getUserMoviesMatrix( data, l_users, l_movies ):
    len_users = len( l_users )
    len_movies = len( l_movies )
    UM = np.zeros( [len_users, len_movies] )
    for i in range(len_users):
        for j in range(len_movies):
            daux = data.loc[ ( data[ data.columns[ 0 ] ]  == l_users[i] ) & ( data[ data.columns[ 1 ] ] == l_movies[j] ) ]
            if daux.empty != True :
                    UM[ i ][ j ] = daux.iloc[0][ daux.columns[ 2 ] ]
    return UM
    

def getRatioMatrix( M ):
    n, m = M.shape
    ratings = 0
    for i in range( n ):
        for j in range( m ):
            if M[i][j] > 0:
                ratings += 1
    sizeArray = n * m
    ratings = ( ratings / float(sizeArray) ) * 100.0
    return ratings



