import pandas as pd


def getFolders( data, l_users, n_folders ):
    users_len = len( l_users )
    user_window = users_len / n_folders
    begin = 0
    end = 0
    folders = []
    for i in range( n_folders ):
        begin = i*user_window
        if (i+1 == folders):
            end =  (i+1)*user_window + (users_len % n_folders)
        else:
             end =  (i+1)*user_window
        users_test = l_users[ begin: end ]
        users_training = [ u for u in l_users if not( u in users_test) ]
        Dtraining = data.loc[ data[ data.columns[ 0 ] ].isin( users_training ) ]
        Dtest = data.loc[ data[ data.columns[ 0 ] ].isin( users_test ) ]
        f_training = '/home/ignacio/Datasets/Samples/bx_folders/bxTrain'+ str(i) +'.csv'
        f_test = '/home/ignacio/Datasets/Samples/bx_folders/bxTest'+ str(i) +'.csv'
        Dtraining.to_csv( f_training, sep=',', index=False )
        Dtest.to_csv( f_test, sep=',', index=False )
        folders.append( (Dtraining, Dtest) ) 
        print( 'begin...{}'.format( begin ) )
        print( 'end...{}'.format( end ) )
    return folders


'''
#preparo la matriz de movielens
print('.........MovieLens........')
data_ml = pd.read_csv('/home/ignacio/Datasets/Samples/ml.csv')
users_ml = list( data_ml['userId'].drop_duplicates() )
movies = list( data_ml['movieId'].drop_duplicates() )
print( data_ml.head() )
print('total users: {}'.format( len( users_ml ) ) )
print('total movies: {}'.format( len( movies ) ) )
print( 'users...{}'.format(len(users_ml)) )
print( 'datos...{}'.format(len(data_ml)) )

folders = getFolders( data_ml, users_ml, 5)
'''

#preparo la matriz de bookcrossing
print('.........Book crossing........')
data_bx = pd.read_csv('/home/ignacio/Datasets/Samples/bx.csv')
users_bx = list( data_bx['User-ID'].drop_duplicates() )
books = list( data_bx['ISBN'].drop_duplicates() )
print( data_bx.head() )
print('total users: {}'.format( len( users_bx ) ) )
print('total books: {}'.format( len( books ) ) )

folders = getFolders( data_bx, users_bx, 5)