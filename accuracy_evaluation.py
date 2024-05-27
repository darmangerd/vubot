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


def compute_tasks_accuracy(df):
    """
    Calculate the number of queries, errors, and accuracy percentage by participant and by task version.
    :param df: DataFrame, input data containing only errors
    # :param grouping_variables: list
    :return: DataFrame, data with calculated accuracy metrics
    """
    queries = df.groupby(by=['ID', 'version'])['response'].size().rename('queries')
    errors = df[df['error']].groupby(by=['ID', 'version'])['response'].count().rename('errors')
    accuracy_ratio = ((1 - errors / queries) * 100).rename('accuracy')

    df_errors = pd.concat([queries, errors, accuracy_ratio], axis=1)
    df_errors.fillna({'errors': 0, 'accuracy': 100}, inplace=True)
    print(
        f"{df_errors}"
    )

    return df_errors


def compute_models_accuracy(df):
    """
    :param df:
    :return: DataFrame, data with calculated accuracy metrics
    """
    queries = df.groupby(by=['ID'])['response'].size().rename('queries')
    errors_speech = df[df['response'] == 'error_speech'].groupby(by=['ID'])['response'].count().rename('errors_speech')
    errors_gesture = df[df['response'] == 'error_gesture'].groupby(by=['ID'])['response'].count().rename('errors_gesture')

    accuracy_gesture = ((1 - errors_gesture / queries) * 100).rename('accuracy_gesture')
    accuracy_speech = ((1 - errors_speech / queries) * 100).rename('accuracy_speech')

    df_errors = pd.concat([queries, errors_gesture, accuracy_gesture, errors_speech, accuracy_speech], axis=1)
    df_errors.fillna({'errors_gesture': 0, 'errors_speech': 0, 'accuracy_gesture': 100, 'accuracy_speech': 100},
                     inplace=True)
    print(
        f"{df_errors}"
    )

    return df_errors


def print_stats_accuracy(df, accuracy_variable, subject):
    """
    Print the mean and standard deviation of the accuracy for a choosen subject.
    :param df: DataFrame of the analysis subject
    :param accuracy_variable: string, name of the df's column containing the accuracy values
    :param subject: string, name of the subject to print
    """
    print(
        f"\nMean accuracy in {subject}: {round(df[accuracy_variable].mean(), 3)} %"
        f"\nStandard deviation of accuracy in {subject}: {round(df[accuracy_variable].std(), 3)} %"
    )


def run_ttest(ttest_df, accuracy_variable1, accuracy_variable2):
    """
    Perform a paired t-test between the two accuracy variables and print the result.
    :param ttest_df: DataFrame, with accuracies values to compare in relative columns
    """
    t_stat, p_value = ttest_rel(ttest_df[accuracy_variable1], ttest_df[accuracy_variable2])
    if p_value < 0.05:
        print(f"\nThe difference is statistically significant: t = {round(t_stat, 3)}, p = {round(p_value, 3)}")
    else:
        print(f"\nThe difference is not statistically significant: t = {round(t_stat, 3)}, p = {round(p_value, 3)}")


def test_accuracy_difference_tasks(df):
    """
    Test for differences in accuracy between the two task versions.
    :param df: DataFrame, data containing only errors
    """
    df_errors = compute_tasks_accuracy(df)
    ttest_df = df_errors.reset_index().pivot(index='ID', columns='version', values='accuracy')

    print_stats_accuracy(ttest_df, 'speech', 'speech commands task')
    print_stats_accuracy(ttest_df, 'keys', 'key presses task')
    run_ttest(ttest_df, 'keys', 'speech')


def test_accuracy_difference_models(df):
    """
    Test for differences in accuracy between the models of speech and gesture recognition.
    :param df: DataFrame, data containing only errors
    """
    df_errors = compute_models_accuracy(df)

    print_stats_accuracy(df_errors, 'accuracy_speech', 'speech recognition model')
    print_stats_accuracy(df_errors, 'accuracy_gesture', 'gesture recognition model')
    run_ttest(df_errors, 'accuracy_gesture', 'accuracy_speech')


def compare_task_accuracies_all_errors():
    print("#### COMPARE TASK VERSIONS: SPEECH COMMANDS VS. KEYS PRESSES - CONSIDERING ALL POSSIBILE ERRORS ####")

    # Load data
    df = accuracy_evaluation_df()

    # Inspect data properties
    print(f"\nTotal queries: {len(df)}"
          f"\nError types: {', '.join([et.split('_')[1] for et in df[df['error']]['response'].unique()])}.")

    # Test for differences in accuracy between the two task versions
    test_accuracy_difference_tasks(df)


def compare_task_accuracies_specific_errors():
    print("#### COMPARE TASK VERSIONS: SPEECH COMMANDS VS. KEYS PRESSES - CONSIDERING ONLY TASK-SPECIFIC ERRORS ####")

    # Load data
    df = accuracy_evaluation_df()
    df = df[~df['error'] | df['response'].isin(['error_speech', 'error_keys'])]

    # Inspect data properties
    print(f"\nTotal queries: {len(df)}"
          f"\nError types: {', '.join([et.split('_')[1] for et in df[df['error']]['response'].unique()])}.")

    # Test for differences in accuracy between the two task versions
    test_accuracy_difference_tasks(df)


def compare_models_accuracies():
    print("#### COMPARE MODELS: SPEECH RECOGNITION VS. GESTURE RECOGNITION ####")

    # Load data
    df = accuracy_evaluation_df()

    # Drop rows containing errors related to speech or gesture
    df = df[~df['error'] | df['response'].isin(['error_speech', 'error_gesture'])]

    # Assign the same value to all responses that were correct
    df.loc[~df['response'].isin(['error_speech', 'error_gesture']), 'response'] = 'correct'

    # Inspect data properties
    print(f"\nTotal queries: {len(df)}"
          f"\nError types: {', '.join([et.split('_')[1] for et in df[df['error']]['response'].unique()])}.")

    # Test for differences in accuracy between the two versions
    test_accuracy_difference_models(df)


def main():
    # Run statistical analysis to compare the overall accuracy of the system in the two task versions
    print(f"\n\n")
    compare_task_accuracies_all_errors()

    # Run statistical analysis to compare the specific accuracy of each task version
    print(f"\n\n")
    compare_task_accuracies_specific_errors()

    # Run statistical analysis to compare the accuracies of speech and gesture recognition models
    print(f"\n\n")
    compare_models_accuracies()




if __name__ == "__main__":
    main()
