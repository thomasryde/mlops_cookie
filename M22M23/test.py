from google.cloud import storage
import pickle
import sklearn
from sklearn.neighbors import *

BUCKET_NAME = "deployment_test2"
MODEL_FILE = "model.pkl"

client = storage.Client()
bucket = client.get_bucket(BUCKET_NAME)
blob = bucket.get_blob(MODEL_FILE)
bob = blob.download_as_string()
my_model = pickle.loads(bob)

def knn_classifier(request):
   """ will to stuff to your request """
   request_json = request.get_json()
   if request_json and 'input_data' in request_json:
         data = request_json['input_data']
         input_data = list(map(int, data.split(',')))
         prediction = my_model.predict([input_data])
         return f'Belongs to class: {prediction}'
   else:
         return 'No input data received'