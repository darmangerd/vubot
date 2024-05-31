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


def compute_task_accuracy(df):
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

    return df_evaluation


def compute_source_error_accuracy(df, error_source):
    """
    Calculate the number of queries, errors, and accuracy percentage by participant and by different error sources.
    :param df: DataFrame, processed input data
    :param error_source: str, name of the recognition model to compute accuracy for
    :return: DataFrame, data with calculated metrics
    """
    # Speech is not contained in keys version and vice-versa; all other error types are in both versions
    if error_source in ['speech', 'keys']:
        df = df[df['version'] == error_source]

    # Drop rows with other errors than the one evaluated
    df = df[~df['error'] | df['response'].isin([f'error_{error_source}'])]

    # Store number of queries, errors, and accuracy percentage in a dataframe
    df_evaluation = compute_model_accuracy(df, error_source)

    return df_evaluation


def run_tasks_ttest(ttest_df, accuracy_variable1, accuracy_variable2):
    """
    Perform a paired t-test between the two accuracy variables and print the result.
    :param ttest_df: DataFrame, with metrics for the evaluation of the two groups
    :param accuracy_variable1: str, name of the column in the df containing the accuracy values of the first group
    :param accuracy_variable2: str, name of the column in the df containing the accuracy values of the second group
    """
    t_stat, p_value = ttest_rel(ttest_df[accuracy_variable1], ttest_df[accuracy_variable2])
    significance = "statistically significant" if p_value < 0.05 else "not statistically significant"
    print(f"\nThe difference between the two groups accuracies is {significance}: "
          f"t = {round(t_stat, 2)}, p = {round(p_value, 2)}")


def run_model_ttest(ttest_df, accuracy_variable, popmean=95):
    """
    Perform a one sample one tailed t-test between the model accuracy and a benchmark (population) value.
    :param ttest_df: DataFrame, with metrics for the evaluation of the group
    :param accuracy_variable: str, name of the column in the df containing the accuracy values
    :param popmean: float, population mean to use as benchmark value for comparison
    """
    t_stat, p_value = ttest_1samp(ttest_df[accuracy_variable], popmean=popmean, alternative='less')
    significance = "statistically lower than" if p_value < 0.05 else "not statistically lower than"
    print(f"\nThe accuracy is {significance} {popmean}% : "
          f"t = {round(t_stat, 2)}, p = {round(p_value, 2)}")


def test_accuracy_difference_tasks(df):
    """
    Test for differences in accuracy between the two task versions.
    :param df: DataFrame, processed input data
    :return: ttest_df: DataFrame, data to compare the two task versions with the t-test
    """
    df_evaluation = compute_task_accuracy(df)
    ttest_df = df_evaluation.reset_index().pivot(index='ID', columns='version', values='accuracy')
    print(f"{ttest_df}")

    print_stats_accuracy(ttest_df, 'speech', 'speech commands task')
    print_stats_accuracy(ttest_df, 'keys', 'key presses task')
    run_tasks_ttest(ttest_df, 'keys', 'speech')
    return ttest_df


def test_accuracy_significance_model(df, model):
    """
    Test for level of accuracy of the recognition model.
    :param df: DataFrame, processed input data
    :param model: str, name of the model to test
    :return: ttest_df: DataFrame, data for the one-sample t-test
    """
    ttest_df = compute_model_accuracy(df, model)
    print(f"{ttest_df}")

    print_stats_accuracy(ttest_df, f'accuracy_{model}', f'{model} recognition model')
    run_model_ttest(ttest_df, f'accuracy_{model}')
    return ttest_df


def compare_task_accuracies_all_errors(df):
    """
    Compare accuracy of the two evaluation task versions.
    :param df: DataFrame, processed input data
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
    Compare accuracy of the two evaluation task versions, only considering error types specific to the tasks.
    :param df: DataFrame, processed input data
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
    :return: DataFrame, data
    """
    # Drop rows containing errors not specific to the model
    df = df[~df['error'] | df['response'].isin([f'error_{model}'])]

    # Assign the same value to all correct responses
    df.loc[~df['response'].isin([f'error_{model}']), 'response'] = 'correct'

    # Print relevant information
    print(f"\nTotal queries: {len(df)}"
          f"\nError types: {', '.join([et.split('_')[1] for et in df[df['error']]['response'].unique()])}.")

    # Compute the evaluation of the accuracy of the single model and save the df used for this evaluation
    df_model_evaluation = test_accuracy_significance_model(df, model)

    return df_model_evaluation


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
    # Immediately remove the queries of the key version as this model was obviously only used in the speech version
    df = df[df['version'] == 'speech']
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
    Explore data properties by displaying different plots.
    :param df: DataFrame, processed input data
    """
    # Define metrics contained into the df that are related to the accuracy evaluation
    metrics = ['queries', 'errors', 'accuracy']

    # Barplots for each metric by participant
    for metric in metrics:
        plot_bar_chart(df, metric)

    # Barplot displaying accuracy by different error sources: recognition models (object, speech, gesture) and keyboard presses version
    error_sources = [et.split('_')[1] for et in df[df['error']]['response'].unique()]
    error_sources.remove('objcolor')

    # Do not consider error related to color in this plot
    not_considered_error = 'color'
    error_sources.remove('color')
    df_filtered = df[df['response'] != f"error_{not_considered_error}"]
    df_filtered = df_filtered[df_filtered['response'] != "error_objcolor"]

    plot_mean_accuracies(df_filtered, error_sources)


def plot_mean_accuracies(df, error_sources):
    """
    Display barplot for the accuracy of each different error source.
    :param df: DataFrame, processed input data
    :param error_sources: list, different sources of error, i.e. recognition models (object, speech, gesture) and keyboard presses
    """
    mean_accuracies, std_accuracies = {}, {}

    for error in error_sources:
        analysis_df = compute_source_error_accuracy(df, error)
        mean_accuracy = analysis_df[f'accuracy_{error}'].mean()
        std_accuracy = analysis_df[f'accuracy_{error}'].std()
        mean_accuracies[error] = mean_accuracy
        std_accuracies[error] = std_accuracy

    # Define colors for the bar plot
    bar_colors = [get_plot_colors_versions()[error] for error in error_sources]

    # Plotting
    fig, ax = plt.subplots()
    ax.bar(mean_accuracies.keys(), mean_accuracies.values(), yerr=std_accuracies.values(), color=bar_colors, capsize=5)
    ax.set_xlabel('Source of Error')
    ax.set_xticks([error for error in mean_accuracies.keys()])
    ax.set_xticklabels([error.capitalize() for error in mean_accuracies.keys()])
    ax.set_ylabel('Mean Accuracy (%)')
    ax.set_yticks(np.arange(0, 120, 20))
    ax.set_title('Mean Accuracy Evaluation')
    plt.show()


def get_plot_colors_versions():
    """
    This functions provides colors specific to the different error source.
    :return: dict, with error source as keys and color shades as values
    """
    return {
        'speech': '#3E2F5B',  # purple
        'object': '#CC3327',  # red
        'gesture': '#4C6FE7',  # blue
        'keys': '#299558'  # green
    }


def plot_bar_chart(df, y_metric):
    """
    Plot bar chart of  for task times data per participant and version.
    :param df: DataFrame, processed input data
    :param y_metric: str, column containing the metric to display on the y-axis
    """
    accuracy_eval_df = compute_task_accuracy(df)
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
        ax.bar(index + i * bar_width, y_values, bar_width, label=version, color=colors[version])

    # Adding labels, title, and legend
    set_accuracy_eval_barplot_labels(ax=ax, metric=y_metric)
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(id)
    ax.legend(title='Version')

    plt.show()


def plot_mean_accuracy_per_task(df, plot_title):
    """
    Create a barplot to display the accuracy of the two tasks versions (speech, keyboard).
    :param df: DataFrame, processed and ready for plotting
    :param plot_title: str, title to display at the top of the plot
    """
    # Compute means and standard deviations
    means = [df['keys'].mean(), df['speech'].mean()]
    stds = [df['keys'].std(), df['speech'].std()]

    fig, ax = plt.subplots(figsize=(8, 7))

    bar_width = 0.75
    tasks = ['keys', 'speech']
    colors = [get_plot_colors_versions()[task] for task in tasks]

    ax.bar(tasks, means, yerr=stds, capsize=5, color=colors, width=bar_width)
    ax.set_xlabel('Task Version')
    ax.set_xticks(tasks)
    ax.set_xticklabels([task.capitalize() for task in tasks])
    ax.set_xlim(-0.6, 1.6)
    ax.set_yticks(np.arange(0, 120, 20))
    ax.set_ylabel('Mean Accuracy (%)')
    ax.set_title(plot_title)

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
    evaluate_gesture_model_accuracy(df)

    # Run analysis to inspect accuracy of speech recognition model
    print(f"\n\n")
    evaluate_speech_model_accuracy(df)

    # Run analysis to inspect accuracy of object recognition model
    print(f"\n\n")
    evaluate_object_model_accuracy(df)


if __name__ == "__main__":
    main()
