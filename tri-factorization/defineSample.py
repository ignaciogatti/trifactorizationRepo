import pandas as pd
import defineData as dd
from sqlalchemy import create_engine


def getSample( data ):
    users, movies = dd.getRandomSampleByMovies(data, 500, 1000)
    dataSample = dd.getSample(data, users, movies)
    return dataSample



#defino muestra del dataset de movielens
data_ml = pd.read_csv('/home/ignacio/Datasets/ml-latest-small/ratings.csv')

print('Working on Movie lens...')

dataSample_ml = getSample( data_ml )
dataSample_ml.to_csv( '/home/ignacio/Datasets/Samples/ml.csv', sep=',', index=False )

#defino muestra del dataset de bookcrossing
data_bx = pd.read_csv('/home/ignacio/Datasets/BX-CSV-Dump/BX-Book-Ratings.csv', sep=';', error_bad_lines=False)
#agrego datos a bookcrossig para manipular dataset similares
print('Working on Book ratings...')

data_bx['timestamp' ] = None
print(data_bx.columns[ 2 ] )
data_bx[ data_bx.columns[ 2 ] ] = data_bx[ data_bx.columns[ 2 ] ].replace( 0.0, 1.0 )
data_bx[ data_bx.columns[ 2 ] ] = data_bx[ data_bx.columns[ 2 ] ] / 2

dataSample_bx = getSample( data_bx )

dataSample_bx.to_csv( '/home/ignacio/Datasets/Samples/bx.csv', sep=',', index=False )

#defino muestra del dataset de netflix
print('Working on Netflix ratings...')

engine = create_engine('mysql+mysqldb://root:@localhost:3306/nfdataset?unix_socket=/opt/lampp/var/mysql/mysql.sock')
connection = engine.connect()
sql_query = 'SELECT t1.* FROM mostRatingsByMovie t1 INNER JOIN (SELECT user_id FROM usersMostRatings WHERE cantidad > 40 ORDER BY rand() LIMIT 500) t2 ON t1.user_id = t2.user_id'
result = connection.execute( sql_query )
dataSample_nf = pd.read_sql(sql_query, engine)

dataSample_nf.to_csv( '/home/ignacio/Datasets/Samples/nf.csv', sep=',', index=False )
