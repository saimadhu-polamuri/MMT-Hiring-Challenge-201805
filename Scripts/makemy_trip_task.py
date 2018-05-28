#! /usr/bin/env Python

# Import required Python packages
import pandas as pd
import pdb

from sklearn import linear_model


def main():
    """

    :return:
    """
    
    # Load train and test datasets
    train_data = pd.read_csv("../Inputs/train.csv")
    test_data = pd.read_csv("../Inputs/test.csv")

    # Create the logistic regression instance
    logreg = linear_model.LogisticRegression()

    # Filter integer headers
    headers = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O']
    integer_headers = ['B', 'C', 'H', 'K', 'N', 'O']

    # Create features and the test datasets
    train_x = train_data[integer_headers]
    train_y = train_data['P']
    test_x = test_data[integer_headers]

    # Fill nan with mean value
    train_x = train_x.fillna(train_x.mean())
    test_x = test_x.fillna(test_x.mean())

    # Train logistic regression model
    logreg.fit(train_x, train_y)
    test_y_predicted = logreg.predict(test_x)

    results_df = pd.DataFrame()
    results_df['id'] = test_data['id']
    results_df['P'] = test_y_predicted
    results_df.to_csv("../Outputs/test_results_with_logistic_v2.csv", index=False)
    print ("File saved...!")

    pdb.set_trace()


if __name__ == "__main__":
    main()
