from flask import Flask, send_from_directory, abort, request
import os
import argparse
debug = True

app = Flask(__name__)
pwd = os.environ['PWD']

argparser = argparse.ArgumentParser()
argparser.add_argument("--type-partitions-provided", type=str, default="iid")
argparser.add_argument("--port", type=int, default=5000)

type_partitions_provided = argparser.parse_args().type_partitions_provided
port_server = argparser.parse_args().port

# Define the base directory where folders are stored
#BASE_DIR = f'{pwd}/partitions-generator/partitions-iid' if type_partitions_provided == 'iid' else f'/partitions-generator/partitions-dirichlet' 

BASE_TEST_DIR = f'{pwd}/total_dataset'

@app.route('/partitions/<idclient>', methods=['GET'])
def get_file_iid(idclient):  
    #Get Query Parameters
    dataset = request.args.get('dataset')
    BASE_DIR = f'{pwd}/partitions-{dataset}-iid'
    print("BASE_DIR", BASE_DIR)
    filename = f'partition_{idclient}.zip'
    if debug:
        print(f'Getting Basedir: {BASE_DIR}')
        print(f'Getting file: {filename}')
    #print(folder_path)
    if os.path.exists(BASE_DIR) and os.path.isdir(BASE_DIR):
        return send_from_directory(BASE_DIR, filename)
    else:
        abort(404, description="Folder not found")

@app.route('/partitions-dirichlet/<idclient>', methods=['GET'])
def get_file_dirichlet(idclient):
    dataset = request.args.get('dataset')
    BASE_DIR = f'{pwd}/partitions-{dataset}-dirichlet'
    filename = f'partition_{idclient}.zip'
    if debug:
        print(f'Getting Basedir: {BASE_DIR}')
        print(f'Getting file: {filename}')
    #print(folder_path)
    if os.path.exists(BASE_DIR) and os.path.isdir(BASE_DIR):
        return send_from_directory(BASE_DIR, filename)
    else:
        abort(404, description="Folder not found")


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=port_server)

