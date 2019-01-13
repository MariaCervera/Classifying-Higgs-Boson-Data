# To generate outputs
from proj1_helpers import *
from helpers import *

def generate_outputs_for_weigths(weights,tx_test,ids_test,file_name):
	"""Function that will generate the submission file given the weights, tx, ids and file name"""
	OUTPUT_PATH = '../../Data/'+file_name+'.csv' 
	y_pred = predict_labels(weights, tx_test)
	create_csv_submission(ids_test, y_pred, OUTPUT_PATH)
	return True