from sklearn.tree import DecisionTreeRegressor
from setup_data import create_data
from utils import save_tree, load_tree

_, _, _, _, trainset, testset, Xtrain, Ytrain, Xtest, Ytest = create_data()

# Fit regression model
tree = DecisionTreeRegressor(max_depth=2)

tree.fit([x.tolist() for x, y in trainset], Ytrain)
print("prediction ....")


save_tree(tree,"regression_tree_model")

tree = load_tree("regression_tree_model")

Ytrain_ = tree.predict([x.tolist() for x, y in trainset])
Ytest_ = tree.predict([x.tolist() for x, y in testset])

print("--- train set ----")
for y, y_ in zip(Ytrain, Ytrain_):
    print(y, y_)
print("--- test set ----")
for y, y_ in zip(Ytest, Ytest_):
    print(y, y_)

