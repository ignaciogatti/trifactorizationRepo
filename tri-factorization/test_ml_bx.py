import pandas as pd
import defineData as dd
import codebookConstruction as cdc
import codebookTransfer as cdt
import metrics as mt

#cargo datasets en memoria
print('.........MovieLens........')
data_ml = pd.read_csv('/home/ignacio/Datasets/Samples/ml.csv')
users_ml = list( data_ml['userId'].drop_duplicates() )
movies = list( data_ml['movieId'].drop_duplicates() )
print( data_ml.head() )
print('total users: {}'.format( len( users_ml ) ) )
print('total movies: {}'.format( len( movies ) ) )

print('.........Book crossing........')
data_bx = pd.read_csv('/home/ignacio/Datasets/Samples/bx.csv')
users_bx = list( data_bx['User-ID'].drop_duplicates() )
books = list( data_bx['ISBN'].drop_duplicates() )
print( data_bx.head() )
print('total users: {}'.format( len( users_bx ) ) )
print('total books: {}'.format( len( books ) ) )


'''
----------Primer prueba---------
matriz auxiliar = movielens
matriz destino = book croosing
user clustering = 50
movie/book clustering = 50
iteraciones = 10
'''
user_clustering = 50
movie_clustering = 50

list_metrics = []
for i in range(10):
    #matriz movielens (Xaux)
    print('\n---------Iteracion {}---------\n'.format(i) )
    print('\nWorking on Movielens\n')
    Dtraining_ml, Dtest_ml, users_training_ml, users_test_ml = mt.defineFolders(data_ml, '/home/ignacio/Datasets/Samples/ml_folders/', i, users_ml )
    dtraining_array_ml = dd.getUserMoviesMatrix( Dtraining_ml, users_training_ml, movies )
    dtest_array_ml = dd.getUserMoviesMatrix(Dtest_ml, users_test_ml, movies)
    #matriz bookcrossing (Xtarget)
    print('\nWorking on Bookcrossing\n')
    Dtraining_bx, Dtest_bx, users_training_bx, users_test_bx = mt.defineFolders(data_bx, '/home/ignacio/Datasets/Samples/bx_folders/', i, users_bx )
    dtraining_array_bx = dd.getUserMoviesMatrix( Dtraining_bx, users_training_bx, books )
    dtest_array_bx = dd.getUserMoviesMatrix(Dtest_bx, users_test_bx, books)
    dtest_array_bx_hide = dd.hideRatings( dtest_array_bx )
    
    #codebook
    print( '\nStarting Codebook algorithm\n' )
    F, S, G = cdc.codebookConstruction( dtraining_array_ml, user_clustering, movie_clustering )
    CB = cdc.defineCodebook( dtraining_array_ml, F, G )
    
    W =cdt.generateMask( dtest_array_bx_hide )
    Ftgt, Gtgt = cdt.codebookTransfer( dtest_array_bx_hide, W, CB )
    
    Xaprox = cdt.getFilledMatrix( dtest_array_bx_hide, W, Ftgt, CB, Gtgt )

    #TODO: redondear ratings

    #genero metricas    
    Xaprox = mt.generateMaskTest( dtest_array_bx, Xaprox )
    mae = mt.getMAE( dtest_array_bx, Xaprox )
    rmse = mt.getRMSE( dtest_array_bx, Xaprox )
    print( '\nMetrics\n' )
    metricas = {'mae':mae, 'rmse':rmse }
    name_serie = 'Run ' + str(i)
    serie_metricas = pd.Series( metricas, name= name_serie )
    list_metrics.append( serie_metricas )
    path_serie = '/home/ignacio/Datasets/Samples/cbt_metricas_parcial'+ str(i) +'.csv'
    serie_metricas.to_csv( path_serie, sep=',', index=False )


df_metrics = pd.DataFrame( list_metrics )
print( df_metrics )
df_metrics.to_csv('/home/ignacio/Datasets/Samples/cbt_metricas.csv', sep=',', index=False )


