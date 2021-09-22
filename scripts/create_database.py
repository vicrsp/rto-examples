import sys, getopt, os
from rto.experiment.db.sqlite import create_rto_db

MEMORY_DATABASE = "file::memory:?cache=shared"

def touch(fname):
    if os.path.exists(fname):
        os.utime(fname, None)
    else:
        open(fname, 'a').close()
    
def create_database(db_path):
    try:
        print(f'Creating database to {db_path}')
        if os.path.exists(db_path):
            os.remove(db_path)
        touch(db_path)
        create_rto_db(db_path)
    except Exception as e:
      print(f'Error creating database: {e}')


if __name__ == '__main__':
    db_name = None
    folder_name = '~'

    try:
        opts, args = getopt.getopt(sys.argv[1:],"hmn:f:",["db=","folder="])
    except getopt.GetoptError:
        print('create_database.py -n <db name> [-f] <folder>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('create_database.py -n <db name> [-f] <folder>')
            sys.exit()
        elif opt in ("-f", "--folder"):
            folder_name = arg
        elif opt in ("-n", "--db"):
            db_name = arg
        elif opt == '-m':
            create_database(MEMORY_DATABASE)
            sys.exit()

    if(db_name is None):
        print('Please provide a valid database name.')
        sys.exit(2)
    db_path = os.path.join(folder_name, f'{db_name}.db')
    create_database(db_path)
