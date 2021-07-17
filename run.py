from utils import *

# While running the file, close current output window to view the consecutive output
if __name__ == "__main__":
    df = convert('SAGES-2-2018-jan-29.txt')  # Specify filename inside quotes
    plotting_peaks(df)
    ai_model_cluster(df, eps=1000, min_samples=3)  # Parameters 'eps' and 'min_samples' can be changed by the user
