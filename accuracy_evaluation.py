import pandas as pd
from scipy.stats import ttest_rel


def import_data(path):
    """
    Import data from a CSV file and reset index.
    :param path: str, path to the CSV file
    :return: DataFrame, imported data
    """
    return pd.read_csv(path, index_col=0).reset_index(drop=True)


def add_error_column(df):
    """
    Create a column indicating if the response was an error or not.
    :param df: DataFrame, input data
    :return: Series, column with boolean values for error responses
    """
    return df['response'].apply(lambda x: x.startswith('error'))


def accuracy_evaluation_df(path="utils/main_evaluation_accuracy.csv"):
    """
    Load and preprocess the accuracy evaluation data.
    :param path: str, path to the CSV file
    :return: DataFrame, preprocessed data with error column added
    """
    df = import_data(path)
    df['error'] = add_error_column(df)

    # Drop columns not needed in evaluation accuracy
    df.drop(columns=['timelog'], inplace=True)
    return df


def compute_accuracy(df):
    """
    Calculate the number of queries, errors, and accuracy percentage by ID and task version.
    :param df: DataFrame, input data containing only errors
    :return: DataFrame, data with calculated accuracy metrics
    """
    queries = df.groupby(by=['ID', 'version'])['response'].size().rename('queries')
    errors = df[df['error']].groupby(by=['ID', 'version'])['response'].count().rename('errors')
    accuracy_ratio = (errors / queries * 100).rename('accuracy')

    df_errors_by_participant_by_version = pd.concat([queries, errors, accuracy_ratio], axis=1)
    df_errors_by_participant_by_version.fillna({'errors': 0, 'accuracy': 100}, inplace=True)

    return df_errors_by_participant_by_version


def print_mean_accuracies(ttest_df):
    """
    Print the mean accuracy for each task version.
    :param ttest_df: DataFrame, pivot table with accuracies for each version
    """
    print(f"\nMean accuracy in keys task: {round(ttest_df['keys'].mean(), 3)} %")
    print(f"Mean accuracy in speech task: {round(ttest_df['speech'].mean(), 3)} %")


def print_std_accuracies(ttest_df):
    """
    Print the standard deviation accuracy for each task version.
    :param ttest_df: DataFrame, pivot table with accuracies for each version
    """
    print(f"\nStandard deviation of accuracy in keys task: {round(ttest_df['keys'].std(), 3)} %")
    print(f"Standard deviation of accuracy in speech task: {round(ttest_df['speech'].std(), 3)} %")


def run_ttest(ttest_df):
    """
    Perform a paired t-test on the accuracies and print the result.
    :param ttest_df: DataFrame, pivot table with accuracies for each task version
    """
    t_stat, p_value = ttest_rel(ttest_df['keys'], ttest_df['speech'])
    if p_value < 0.05:
        print(f"\nThe difference is statistically significant: t = {round(t_stat, 3)}, p = {round(p_value, 3)}")
    else:
        print(f"\nThe difference is not statistically significant: t = {round(t_stat, 3)}, p = {round(p_value, 3)}")


def test_accuracy_difference(df):
    """
    Test for differences in accuracy between the two versions.
    :param df: DataFrame, data containing only errors
    """
    df_errors_by_participant_by_version = compute_accuracy(df)
    ttest_df = df_errors_by_participant_by_version.reset_index().pivot(index='ID', columns='version', values='accuracy')

    print_mean_accuracies(ttest_df)
    print_std_accuracies(ttest_df)
    run_ttest(ttest_df)


def main():
    # Load data
    df = accuracy_evaluation_df()

    # Test for differences in accuracy between the two versions
    test_accuracy_difference(df)


if __name__ == "__main__":
    main()
