from sklearn import tree,svm
from sklearn import ensemble
from sklearn.ensemble import GradientBoostingClassifier


models = 
{
	"decision_tree_gini": tree.DecisionTreeClassifier(criterion="gini"),

 	"svm":svm.SVC(),

 	"rf":ensemble.RandomForestClassifier(),
 	
 	"gb":GradientBoostingClassifier()

}