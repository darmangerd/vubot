import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
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


def accuracy_evaluation_df(path=r"main_evaluation_accuracy.csv"):
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
    return ttest_df


def test_accuracy_significance_model(df, model):
    """
    Test for level of accuracy of the recognition model.
    :param df: DataFrame, processed input data
    :param model: str, name of the model to test
    """
    df_evaluation = compute_model_accuracy(df, model)

    print_stats_accuracy(df_evaluation, f'accuracy_{model}', f'{model} recognition model')
    run_model_ttest(df_evaluation, f'accuracy_{model}')
    return df_evaluation


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
    analysis_df = test_accuracy_difference_tasks(df)
    plot_mean_accuracy_per_task(analysis_df, plot_title='Mean Accuracy by Task Version (All Errors)')


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
    analysis_df = test_accuracy_difference_tasks(df)
    plot_mean_accuracy_per_task(analysis_df, plot_title='Mean Accuracy by Task Version (Specific Errors)')


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
    df_model_evaluation = test_accuracy_significance_model(df, model)

    return df_model_evaluation


def evaluate_gesture_model_accuracy(df):
    """
    :param df: DataFrame, processed input data
    Evaluate accuracy of the gesture recognition model.
    """
    print("#### EVALUATE MODEL: GESTURE RECOGNITION ####")
    df_model_evaluation = evaluate_model_accuracy(df=df, model='gesture')
    return df_model_evaluation


def evaluate_speech_model_accuracy(df):
    """
    :param df: DataFrame, processed input data
    Evaluate accuracy of the speech recognition model.
    """
    print("#### EVALUATE MODEL: SPEECH RECOGNITION ####")
    df_model_evaluation = evaluate_model_accuracy(df=df, model='speech')
    return df_model_evaluation


def evaluate_object_model_accuracy(df):
    """
    :param df: DataFrame, processed input data
    Evaluate accuracy of the object recognition model.
    """
    print("#### EVALUATE MODEL: OBJECT RECOGNITION ####")
    df_model_evaluation = evaluate_model_accuracy(df=df, model='object')
    return df_model_evaluation


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


def get_plot_colors_versions():
    return ['#299558', '#3E2F5B', '#E84427']


def plot_bar_chart(df, y_metric):
    """
    Plot bar chart of  for task times data per participant and version.
    :param df: DataFrame, processed input data
    :param y_metric: str, column containing the metric to display on the y-axis
    """
    accuracy_eval_df = compute_tasks_accuracy(df)
    plot_df = accuracy_eval_df.reset_index().pivot(index='ID', columns='version', values=y_metric)
    accuracy_eval_barplot(plot_df, y_metric)


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


def accuracy_eval_barplot(df, y_metric):
    """
    Create barplot for task times data per participant and version.
    :param df: DataFrame, processed and ready for plotting
    :param y_metric: str, column containing the metric to display on the y-axis
    """
    id = df.index.unique()
    versions = df.columns.unique()
    index = np.arange(len(id))

    fig, ax = plt.subplots(figsize=(10, 6))
    bar_width = 0.35
    colors = get_plot_colors_versions()

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


def plot_mean_accuracy_per_task(df, plot_title):
    # Compute means and standard deviations
    means = [df['keys'].mean(), df['speech'].mean()]
    stds = [df['keys'].std(), df['speech'].std()]

    fig, ax = plt.subplots(figsize=(8, 7))

    bar_width = 0.75
    labels = ['Keys', 'Speech']
    colors = get_plot_colors_versions()

    ax.bar(labels, means, yerr=stds, capsize=5, color=colors, width=bar_width)
    ax.set_xlabel('Task Version')
    ax.set_yticks(np.arange(0, 120, 20))
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylabel('Mean Accuracy %')
    # fig.suptitle(plot_title)
    ax.set_title(plot_title)

    plt.show()


def plot_models_accuracies(dfs_to_concat):
    """

    """
    df = pd.concat(dfs_to_concat, axis=1)
    accuracy_cols = df.filter(regex='^accuracy').columns.tolist()
    models = [acc.split('_')[1].capitalize() for acc in accuracy_cols]

    fig, ax = plt.subplots(figsize=(9, 7))
    bar_width = 0.35
    # colors = cm.get_cmap('Accent', len(models)).colors
    colors = get_plot_colors_versions()

    # Create bars for each version and each participant
    for i, acc in enumerate(accuracy_cols):
        y_values = df[acc]
        y_values_mean = y_values.mean()
        y_values_std = y_values.std()
        ax.bar(i/2, y_values_mean, yerr=y_values_std, width=bar_width, label=acc, color=colors[i])

    # Adding labels, title, and legend
    ax.set_xticks([i/2 for i, _ in enumerate(models)])
    ax.set_xticklabels(labels=models)
    ax.set_xlabel('Recognition Models')
    ax.set_xlim(-0.3, 1.3)
    ax.set_yticks(np.arange(0, 120, 20))
    ax.set_ylabel('Mean Accuracy %')
    ax.set_title('Accuracy of the Recognition Models')

    plt.show()


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
    df_gesture_model_eval = evaluate_gesture_model_accuracy(df)

    # Run analysis to inspect accuracy of speech recognition model
    print(f"\n\n")
    df_speech_model_eval = evaluate_speech_model_accuracy(df)

    # Run analysis to inspect accuracy of object recognition model
    print(f"\n\n")
    df_object_model_eval = evaluate_object_model_accuracy(df)

    # Display model evaluations results
    plot_models_accuracies([df_gesture_model_eval, df_speech_model_eval, df_object_model_eval])


if __name__ == "__main__":
    main()
