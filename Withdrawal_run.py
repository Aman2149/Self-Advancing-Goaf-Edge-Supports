from Withdrawal_utils import *

if __name__ == '__main__':
    n_sages, featureDict_gbm = getInput()
    prediction = get_prediction(n_sages, featureDict_gbm)
    print(f'Your Convergence value for SAGES={n_sages} is {prediction}')
