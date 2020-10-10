import numpy as np

from setup_data import create_data, retreive_db_data
from utils import load_tree

# _, _, _, _, trainset, testset, Xtrain, Ytrain, Xtest, Ytest, data = create_data()
# tree = load_tree("../regression_tree_model")
#
# # X = np.random.random_sample((1,19))
# Y = tree.predict(Xtest)
# print(Y)

data, _ = retreive_db_data()
del data["CreatedByUserId"]
# del data["UserId"]
del data["CreditScore"]
del data["CreationDate"]
del data["Id"]

# noisy data
del data["Income"]
del data["JobClass"]
del data["EmploymentType"]
del data["SocialStability"]
del data["SocialExposure"]
del data["SocialQuality"]
print("ok")