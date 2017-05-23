import numpy as np
import defineData as dd
import codebookTransfer as cdt


def generateMaskTest(Xtest, Xp):
    Mtest = cdt.generateMask( Xtest )
    Mp = Xp * Mtest
    return Mp


def getMAE(Xtest, Xp):
    Xp = generateMaskTest( Xtest, Xp )
    Xresult = np.abs(Xtest - Xp)
    mae = np.sum(Xresult)
    n_ratings = np.count_nonzero( Xtest )
    return ( mae/n_ratings )


def getRMSE(Xtest, Xp):
    Xp = generateMaskTest( Xtest, Xp )
    Xresult = Xtest - Xp
    Xresult = Xresult**2
    rmse = np.sum(Xresult)
    n_ratings = np.count_nonzero( Xtest )
    rmse = rmse / n_ratings
    return ( np.sqrt( rmse ) )


def defineFolders( data, path_file, iteration, users ):

    Dtraining, Dtest, users_training, users_test = dd.getTestSet( data, users, 200 )
    path_training = path_file + 'Train' + str( iteration ) + '.csv'
    path_test = path_file + 'Test' + str( iteration ) + '.csv'
    Dtraining.to_csv( path_training, sep=',', index=False )
    Dtest.to_csv( path_test, sep=',', index=False )
    return Dtraining, Dtest, users_training, users_test



'''
#preparo la matriz de netflix
print('.........Netflix..........')
data_nf = pd.read_csv('/home/ignacio/Datasets/Samples/nf.csv')
users_nf = list( data_nf['user_id'].drop_duplicates() )
movies_nf = list( data_nf['movie_id'].drop_duplicates() )
print(data_nf.head() )
print('total users: {}'.format( len( users_nf ) ) )
print('total movies: {}'.format( len( movies_nf ) ) )

'''