import pandas as pd
import numpy as np
import re
from os import listdir
from os.path import isfile, join
import MySQLdb
import datetime
from sqlalchemy import create_engine

path = '/home/ignacio/Datasets/nf-dataset/training_set/'
pathfile = '/home/ignacio/Datasets/nf-dataset/training_set/mv_0000001.txt'


def parsedata():

    engine = create_engine('mysql+mysqldb://root:@localhost:3306/nfdataset?unix_socket=/opt/lampp/var/mysql/mysql.sock')
    files = [f for f in listdir(path) if isfile( join( path, f ) ) ]
    index_files = np.arange( len( files ) )
    np.random.shuffle( index_files )
    index_files = index_files[:10]
    nf_dataset = pd.DataFrame()
    for f in files:
        data = pd.read_csv(join(path, f), names=['user_id','rating','date'], parse_dates=['date'] )
        m_id = data.iloc[0][ data.columns[0] ]
        m_id = re.sub(':', '', m_id )
        m_id = int(m_id)
        data.drop( data.index[ 0 ], inplace = True )
        data['movie_id'] = m_id
        cols = data.columns.tolist()
        cols = cols[0:1] + cols[-1:] + cols[1:-1]
        data = data[ cols ]
        data = data.set_index(['user_id','movie_id'])
#        nf_dataset = pd.concat([nf_dataset, data], ignore_index = True )
        data.to_sql('ratings', engine, if_exists='append')

        print(" Working with " + f)
        f=open('/home/ignacio/Datasets/lastFileRead.txt', 'w')
        f.write( str(f) )
        f.close()
    engine.dispose()        
    return 0



def conectDB():
    db = MySQLdb.connect(host='127.0.0.1', user='root', passwd='', db='nfdataset', unix_socket='/opt/lampp/var/mysql/mysql.sock', port=3306)
    
    cursor = db.cursor()
    numrows = cursor.execute( "SHOW TABLES" )
    print("rows...{}".format( numrows ) )
    rows = cursor.fetchall()
    for row in rows:
        for col in row:
            print("{}, ".format(col))
        print( "\n" )
    fecha = datetime.date(2016, 3, 28)
    cursor.execute( "INSERT INTO ratings (user_id, movie_id, rating, date) VALUES (%s,%s,%s,%s)", (3, 5, 3.0,fecha) )
    
    db.commit()
    
    cursor.close()
    db.close()
    return 0


#pcargo los datos a la base de datos

data = parsedata()
if data == 0:
    print('proceso finalizado con exito')
else:
    print('error durante la ejecucion')
