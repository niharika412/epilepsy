import os
import joblib
import pandas as pd
from sklearn import metrics
import config
import argparse
import model_dispatcher

def run(fold, model):

	df=pd.read_csv(config.TRAINING_FILE)

	df_train=df[df.kfold != fold].reset_index(drop=True)

	df_valid= df[df.kfold == fold].reset_index(drop=True)

	x_train=df_train.drop("activity",axis=1).values
	y_train=df_train.activity.values

	x_valid=df_valid.drop("activity",axis=1).values
	y_valid=df_valid.activity.values
	#print("debug")

	clf= model_dispatcher.models[model]


	clf.fit(x_train,y_train)

	preds=clf.predict(x_valid)

	accuracy= metrics.accuracy_score(y_valid,preds)
	print("Fold=",fold,"\t accuracy",accuracy)

	joblib.dump(
		clf,os.path.join(config.MODEL_OUTPUT, f"dt_{fold}.bin"))

if __name__ == "__main__":

	parser= argparse.ArgumentParser()

	parser.add_argument(
		'--fold',type=str)
	parser.add_argument(
 	"--model",type=str)

	args = parser.parse_args()

	if args.fold=='all':
		run(fold=0,model=args.model)
		run(fold=1,model=args.model)
		run(fold=2,model=args.model)
		run(fold=3,model=args.model)
		run(fold=4,model=args.model)
	else:
		folds=int(args.fold)

		run(fold=folds, model=args.model)


