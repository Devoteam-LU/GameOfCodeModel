import pandas as pd
import pyodbc
import torch
import torch.utils.data as data_utils
import numpy as np


def retreive_db_data():
    conn = pyodbc.connect(DRIVER='{ODBC Driver 17 for SQL Server}',
                          SERVER='devohackaton.database.windows.net',
                          DATABASE='privilege',
                          UID='aadmin',
                          PWD='9rTuUjFcssBUGws2')

    cursor = conn.cursor()

    bdd_data = pd.read_sql_query('SELECT * FROM [privilege].[dbo].[userdetailtrains]', conn)
    credit_scores = pd.read_sql_query(
        'SELECT [UserId],[Amount],[Probability] FROM [privilege].[dbo].[RepaymentProbabilities]', conn)
    return bdd_data, credit_scores


def create_data():
    bdd_data, credit_scores = retreive_db_data()

    # merge user data and creditscore
    data = []
    for index, row in credit_scores.iterrows():
        client_detail = bdd_data[bdd_data.UserId == row["UserId"]]
        new_row = [client_detail.iloc[0][col] for col in bdd_data.head()]
        new_row += [row["Amount"], row["Probability"]]
        data.append(new_row)

    data = pd.DataFrame(data, columns=list(bdd_data.head()) + ["Amount", "Probability"])
    del data["CreatedByUserId"]
    del data["UserId"]
    del data["CreditScore"]
    del data["CreationDate"]
    del data["Id"]
    del data["FirstName"]
    del data["LastName"]

    # noisy data
    del data["Income"]
    del data["JobClass"]
    del data["EmploymentType"]
    del data["SocialStability"]
    del data["SocialExposure"]
    del data["SocialQuality"]

    pd.set_option("display.max_rows", None, "display.max_columns", None)
    # print(data)
    # data = data.iloc[:3]

    # preprocessing the data
    data = data.replace("M", 0)
    data = data.replace("F", 1)
    data = data.replace(True, 1)
    data = data.replace(False, 0)

    dff = data.drop('Probability', axis=1)
    mean = dff.mean().values
    std = dff.std().values

    # compute mean and variance of the data

    ratio = 0.75
    n_train = int(ratio * data.shape[0])

    input_size = data.shape[1] - 1

    train_df = data.iloc[:n_train]
    test_df = data.iloc[n_train:]

    def create_dataset(dataframe):
        y = torch.tensor(dataframe['Probability'].values.astype(np.float32))
        x = torch.tensor(dataframe.drop('Probability', axis=1).values.astype(np.float32))
        dataset = data_utils.TensorDataset(x, y)
        return dataset

    trainset = create_dataset(train_df)
    testset = create_dataset(test_df)

    Ytrain = [y.tolist() for x, y in trainset]
    Ytest = [y.tolist() for x, y in testset]
    Xtrain = [x.tolist() for x, y in trainset]
    Xtest = [x.tolist() for x, y in testset]

    return mean, std, n_train, input_size, trainset, testset, Xtrain, Ytrain, Xtest, Ytest, data
