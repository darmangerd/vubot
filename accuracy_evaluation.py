import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from scipy.stats import ttest_1samp


def import_data(path):
    """
    Import data recorded during the evaluation trials from the CSV file.
    :param path: str, path to the CSV file
    :return: DataFrame, imported data
    """
    return pd.read_csv(path, index_col=0).reset_index(drop=True)


def add_error_column(df):
    """
    Create a column indicating if the value on the same row and in the response column was an error or not.
    :param df: DataFrame, input data
    :return: Series, column with boolean values informative of the error responses
    """
    return df['response'].apply(lambda x: x.startswith('error'))


def accuracy_evaluation_df(path=r"utils/main_evaluation_accuracy.csv"):
    """
    Load and preprocess the accuracy evaluation data.
    :param path: str, path to the CSV file of the data recorded during trials
    :return: DataFrame, preprocessed data with error column added
    """
    df = import_data(path)
    df['error'] = add_error_column(df)
    df.drop(columns=['timelog'], inplace=True)
    return df


def compute_tasks_accuracy(df):
    """
    Calculate the number of queries, errors, and accuracy percentage by participant and by task version.
    :param df: DataFrame, processed input data
    :return: DataFrame, data with calculated metrics
    """
    queries = df.groupby(by=['ID', 'version'])['response'].size().rename('queries')
    errors = df[df['error']].groupby(by=['ID', 'version'])['response'].count().rename('errors')
    accuracy_ratio = ((1 - errors / queries) * 100).rename('accuracy')

    df_evaluation = pd.concat([queries, errors, accuracy_ratio], axis=1)
    df_evaluation.fillna({'errors': 0, 'accuracy': 100}, inplace=True)
    print(f"{df_evaluation}")

    return df_evaluation


def compute_model_accuracy(df, model):
    """
    Calculate the number of queries, errors, and accuracy percentage by participant and by task version.
    :param df: DataFrame, processed input data

    :param model: str, name of the recognition model to compute accuracy for
    :return: DataFrame, data with calculated metrics
    """
    queries = df.groupby(by=['ID'])['response'].size().rename('queries')
    errors = df[df['response'] == f'error_{model}'].groupby(by=['ID'])['response'].count().rename(f'errors_{model}')

    accuracy = ((1 - errors / queries) * 100).rename(f'accuracy_{model}')

    df_evaluation = pd.concat([queries, errors, accuracy], axis=1)
    df_evaluation.fillna({f'errors_{model}': 0, f'accuracy_{model}': 100},inplace=True)
    print(f"{df_evaluation}")

    return df_evaluation


def print_stats_accuracy(df, accuracy_variable, subject):
    """
    Print the mean and standard deviation of the accuracy for a chosen subject.
    :param df: DataFrame, with metrics for the evaluation
    :param accuracy_variable: string, name of the df's column containing the accuracy values
    :param subject: string, name of the subject to print
    """
    print(
        f"\nMean accuracy in {subject}: {round(df[accuracy_variable].mean(), 2)}%"
        f"\nStandard deviation of accuracy in {subject}: {round(df[accuracy_variable].std(), 2)}%"
    )


def run_tasks_ttest(ttest_df, accuracy_variable1, accuracy_variable2):
    """
    Perform a paired t-test between the two accuracy variables and print the result.
    :param ttest_df: DataFrame, with metrics for the evaluation
    :param accuracy_variable1: str, name of the column in the df containing the accuracy values of the first group
    :param accuracy_variable2: str, name of the column in the df containing the accuracy values of the second group
    """
    t_stat, p_value = ttest_rel(ttest_df[accuracy_variable1], ttest_df[accuracy_variable2])
    significance = "statistically significant" if p_value < 0.05 else "not statistically significant"
    print(f"\nThe difference between the two groups accuracies is {significance}: t = {round(t_stat, 2)}, p = {round(p_value, 2)}")


def run_model_ttest(ttest_df, accuracy_variable, popmean=95):
    """
    Perform a one sample t-test between the model accuracy and a benchmark (population) value.
    :param ttest_df: DataFrame, with metrics for the evaluation
    :param accuracy_variable: str, name of the column in the df containing the accuracy values
    :param popmean: float, population mean to use as benchmark value for comparison
    """
    t_stat, p_value = ttest_1samp(ttest_df[accuracy_variable], popmean=popmean, alternative='less')
    significance = "statistically lower than" if p_value < 0.05 else "not statistically lower than"
    print(f"\nThe accuracy is {significance} {popmean}% : t = {round(t_stat, 2)}, p = {round(p_value, 2)}")


def test_accuracy_difference_tasks(df):
    """
    Test for differences in accuracy between the two task versions.
    :param df: DataFrame, processed input data

    """
    df_evaluation = compute_tasks_accuracy(df)
    ttest_df = df_evaluation.reset_index().pivot(index='ID', columns='version', values='accuracy')

    print_stats_accuracy(ttest_df, 'speech', 'speech commands task')
    print_stats_accuracy(ttest_df, 'keys', 'key presses task')
    run_tasks_ttest(ttest_df, 'keys', 'speech')


def test_accuracy_significance_model(df, model):
    """
    Test for level of accuracy of the recognition model.
    :param df: DataFrame, processed input data

    :param model: str, name of the model to test
    """
    df_evaluation = compute_model_accuracy(df, model)

    print_stats_accuracy(df_evaluation, f'accuracy_{model}', f'{model} recognition model')
    run_model_ttest(df_evaluation, f'accuracy_{model}')


def compare_task_accuracies_all_errors(df):
    """
    :param df: DataFrame, processed input data
    Compare accuracy of the two evaluation task versions.
    """
    print("#### COMPARE TASK VERSIONS: SPEECH COMMANDS VS. KEYS PRESSES - CONSIDER ALL POSSIBILE ERRORS ####")

    # Print relevant information
    print(f"\nTotal queries: {len(df)}"
          f"\nError types: {', '.join([et.split('_')[1] for et in df[df['error']]['response'].unique()])}.")

    # Test for differences in accuracy between the two task versions
    test_accuracy_difference_tasks(df)


def compare_task_accuracies_specific_errors(df):
    """
    :param df: DataFrame, processed input data
    Compare accuracy of the two evaluation task versions, only considering error types specific to the tasks.
    """
    print("#### COMPARE TASK VERSIONS: SPEECH COMMANDS VS. KEYS PRESSES - CONSIDER ONLY TASK-SPECIFIC ERRORS ####")

    # Filter error responses to those specific for the task versions
    df = df[~df['error'] | df['response'].isin(['error_speech', 'error_keys'])]

    # Print relevant information
    print(f"\nTotal queries: {len(df)}"
          f"\nError types: {', '.join([et.split('_')[1] for et in df[df['error']]['response'].unique()])}.")

    # Test for differences in accuracy between the two task versions
    test_accuracy_difference_tasks(df)


def evaluate_model_accuracy(df, model):
    """
    Evaluate accuracy of a single recognition model.
    :param df: DataFrame, processed input data
    :param model: str, name of the model to evaluate
    """
    # if model == 'object':
    #     df['response'] = df['response'].apply(lambda x: 'error_object' if x == 'error_objcolor' else x)

    # Drop rows containing errors not specific to the model
    df = df[~df['error'] | df['response'].isin([f'error_{model}'])]

    # Assign the same value to all correct responses
    df.loc[~df['response'].isin([f'error_{model}']), 'response'] = 'correct'

    # Print relevant information
    print(f"\nTotal queries: {len(df)}"
          f"\nError types: {', '.join([et.split('_')[1] for et in df[df['error']]['response'].unique()])}.")

    # Compute statistics for the accuracy of the model for differences in accuracy between the two versions
    test_accuracy_significance_model(df, model)


def evaluate_gesture_model_accuracy(df):
    """
    :param df: DataFrame, processed input data
    Evaluate accuracy of the gesture recognition model.
    """
    print("#### EVALUATE MODEL: GESTURE RECOGNITION ####")
    evaluate_model_accuracy(df=df, model='gesture')


def evaluate_speech_model_accuracy(df):
    """
    :param df: DataFrame, processed input data
    Evaluate accuracy of the speech recognition model.
    """
    print("#### EVALUATE MODEL: SPEECH RECOGNITION ####")
    evaluate_model_accuracy(df=df, model='speech')


def evaluate_object_model_accuracy(df):
    """
    :param df: DataFrame, processed input data
    Evaluate accuracy of the object recognition model.
    """
    print("#### EVALUATE MODEL: OBJECT RECOGNITION ####")
    evaluate_model_accuracy(df=df, model='object')


def explore_dataset(df):
    """
    ...
    :param df: DataFrame, processed input data
    """
    # Define metrics contained into the df that are related to the accuracy evaluation
    metrics = ['queries', 'errors', 'accuracy']

    # Barplots for each metric
    for metric in metrics:
        plot_bar_chart(df, metric)


def plot_bar_chart(df, y_metric):
    """
    Plot bar chart of  for task times data per participant and version.
    :param df: DataFrame, processed input data
    :param y_metric: str, column containing the metric to display on the y-axis
    """
    accuracy_eval_df = compute_tasks_accuracy(df)
    plot_df = accuracy_eval_df.reset_index().pivot(index='ID', columns='version', values=y_metric)
    accuracy_eval_barplot(plot_df, y_metric)


def accuracy_eval_barplot(df, y_metric):
    """
    Create barplot for task times data per participant and version.
    :param df: DataFrame, processed and ready for plotting
    :param y_metric: str, column containing the metric to display on the y-axis
    """
    id = df.index.unique()
    versions = df.columns.unique()
    print(
        f"{versions = }"
    )
    index = np.arange(len(id))

    fig, ax = plt.subplots(figsize=(10, 6))
    bar_width = 0.35
    colors = ['#1D2F6F', '#8390FA']
    # colors = ['#1c6f1c', '#83fa83']  # green shade

    # Create bars for each version and each participant
    for i, version in enumerate(versions):
        y_values = df[f"{version}"]
        ax.bar(index + i * bar_width, y_values, bar_width, label=version, color=colors[i])

    # Adding labels, title, and legend
    set_accuracy_eval_barplot_labels(ax=ax, metric=y_metric)
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(id)
    ax.legend(title='Version')

    plt.show()


def set_accuracy_eval_barplot_labels(ax, metric):
    """
    Set axis labels for accuracy evaluation barplot.
    :param ax: Axes object, the plot axes
    :param metric: str, the metric to set labels for
    """
    ax.set_xlabel('Participant')

    if metric == 'errors':
        ax.set_ylabel('Number of Errors')
    elif metric == 'accuracy':
        ax.set_ylabel('Accuracy Rate')
    elif metric == 'queries':
        ax.set_ylabel('Number of Queries')

    if metric == 'errors':
        ax.set_title("Errors in Bot's Responses")
    elif metric == 'accuracy':
        ax.set_title("Accuracy of Bot's Responses")
    elif metric == 'queries':
        ax.set_title('Number of Queries Needed to Finish Task')


def main():
    # Load the evaluation data
    df = accuracy_evaluation_df()

    # Perform general data inspection
    explore_dataset(df)

    # Run statistical analysis to compare the overall accuracy of the system in the two task versions
    print(f"\n\n")
    compare_task_accuracies_all_errors(df)

    # Run statistical analysis to compare the specific accuracy of each task version
    print(f"\n\n")
    compare_task_accuracies_specific_errors(df)

    # Run analysis to inspect accuracy of gesture recognition model
    print(f"\n\n")
    evaluate_gesture_model_accuracy(df)

    # Run analysis to inspect accuracy of speech recognition model
    print(f"\n\n")
    evaluate_speech_model_accuracy(df)

    # Run analysis to inspect accuracy of object recognition model
    print(f"\n\n")
    evaluate_object_model_accuracy(df)


if __name__ == "__main__":
    main()
