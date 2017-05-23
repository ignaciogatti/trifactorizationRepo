import pandas as pd
from surprise import KNNBasic
from surprise import Dataset, Reader
from surprise import accuracy
from os.path import expanduser
from os import listdir
from os.path import isfile, join
import re


def getFolds( dir_path ):
    files = [f for f in listdir(dir_path) if isfile( join( dir_path, f ) ) ]
    
    n_folders = len( files )/2
    folds_file = []
    
    for i in range( n_folders ):
        train_re = re.compile( 'Train' + str(i) + '.csv' )
        test_re = re.compile( 'Test' + str(i) + '.csv' )
        train_file = [f for f in files if re.search(train_re, f)!=None ][ 0 ]
        train_file = dir_path + train_file
        test_file = [ f for f in files if re.search(test_re, f)!=None ][ 0 ]
        test_file = dir_path + test_file
        folds_file.append( (train_file, test_file) )
    return folds_file
        


dir_path_films = expanduser('/home/ignacio/Datasets/Samples/ml_folders/')
dir_path_books = expanduser('/home/ignacio/Datasets/Samples/bx_folders/')

reader_films = Reader( line_format='user item rating timestamp', sep=',', skip_lines=1 )
reader_books = Reader( line_format='user item rating timestamp', sep=',', skip_lines=1 )


folds_file = getFolds( dir_path_books )
data_films = Dataset.load_from_folds( folds_file, reader_books )

algo = KNNBasic()

predictions = None
metrics = []
i = 0
for training_set, test_set in data_films.folds():
    
    algo.train( training_set )
    predictions = algo.test( test_set )
    
    rmse = accuracy.rmse( predictions, verbose=True )
    mae = accuracy.mae( predictions, verbose=True )
    metricas = { 'mae':mae, 'rmse':rmse }
    name_serie = 'Run ' + str(i)
    #guardo las matricas
    serie_metricas = pd.Series( metricas, name= name_serie )
    metrics.append( serie_metricas )
    path_serie = '/home/ignacio/Datasets/Samples/uu_metricas_parcial'+ str(i) +'.csv'
    serie_metricas.to_csv( path_serie, sep=',', index=False )
    i += 1

df_metrics = pd.DataFrame( metrics )
print( df_metrics )
df_metrics.to_csv('/home/ignacio/Datasets/Samples/uu_metricas.csv', sep=',', index=False )
