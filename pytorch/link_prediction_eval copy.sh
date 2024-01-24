relation=$1

conda activate py311
python evaluate.py $relation 
python transR_eval.py $relation
python transE_eval.py $relation
python transH_eval.py $relation
python transD_eval.py $relation