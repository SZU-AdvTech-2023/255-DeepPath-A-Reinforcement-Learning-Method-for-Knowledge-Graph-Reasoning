relation=$1
# relation = "concept_athletehomestadium"
conda activate py311
python evaluate.py $relation 
python transR_eval.py $relation
python transE_eval.py $relation
