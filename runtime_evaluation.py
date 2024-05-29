import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_rel

def import_data(path):
    """
    Import data recorded during the evaluation trials from the CSV file.
    :param path: str, path to the CSV file
    :return: DataFrame, imported data
    """
    return pd.read_csv(path)

def preprocess_data(df):
    """
    Preprocess the runtime evaluation data to remove rows with errors.
    :param df: imported evaluation DataFrame
    :return: DataFrame, preprocessed data without error rows and unnecessary columns
    """
    df.dropna(inplace=True)
    df.drop(columns=['Unnamed: 0', 'response'], inplace=True)
    return df

def prep_task_time_ttest_df(df,verbose=True):
    """
    Prepares task time data for t-test.
    :param df: imported task times DataFrame
    :return: DataFrame, columns for each version with the task of each participant in the rows
    """
    ttest_df = df.pivot(index='Participant', columns='Version', values='Task Length Decimal')
    if verbose:
        print(f"\n{ttest_df}")
    return ttest_df

def completed_ratio(df, version):
    """
    Calculates and prints the number of participants that completed the tasks of a specified version.
    :param df: imported task times DataFrame
    :param version: str, version of the task, either 'Keys' or 'Speech'
    """
    df = df[df['Version'] == version]['Task Completed']
    completed = df.sum()
    print(f"{completed}/{len(df)} participants completed the {version.lower()} version task")

def create_evaluation_df(df, verbose=True):
    """
    Creates stats DataFrame for evaluation.
    :param df: imported DataFrame
    :param verbose: prints a table with the mean and std of the query time per participant and version
    :return: DataFrame, includes ID, task, version, mean, and std of the query time
    """
    df = df.groupby(['ID','task','version'])
    summary = df['timelog'].agg(['mean', 'std']).reset_index()
    if verbose:
        print(summary)
    return summary

def prep_version_ttest_df(df, task, verbose=True):
    """
    Prepares runtime data for t-test.
    :param df: evaluation DataFrame
    :param task: str, task (either 'object' or 'color')
    :return: DataFrame, columns for each version with the mean runtime of each query per participant.
    """
    df = df[df['task'] == task]
    ttest_df = df.pivot(index='ID', columns='version', values='mean')
    if verbose:
        print(f"\n{ttest_df}")
    return ttest_df

def runtime_evaluation_df(path=r"utils/main_evaluation_accuracy.csv"):
    """
    Load and preprocess the runtime evaluation data.
    :param path: str, path to the CSV file of the data recorded during trials
    :return: DataFrame, preprocessed data with error column added
    """
    df = import_data(path)
    df.dropna(inplace=True)
    df.drop(columns=['Unnamed: 0', 'response'], inplace=True)
    return df

def print_stats_runtime(df, runtime_variable, subject, unit):
    """
    Print the mean and standard deviation of the runtime for a chosen subject.
    :param df: DataFrame, with metrics for the evaluation
    :param runtime_variable: str, name of the df's column containing the mean runtime values
    :param subject: str, name of the subject to print
    :param unit: str, unit of measurement
    """
    print(
        f"\nMean runtime in {subject}: {round(df[runtime_variable].mean(), 2)}{unit}"
        f"\nStandard deviation of runtime in {subject}: {round(df[runtime_variable].std(), 2)}{unit}"
    )

def run_paired_ttest(ttest_df, variable1, variable2):
    """
    Perform a paired t-test between the two runtime variables and print the result.
    :param ttest_df: DataFrame, with metrics for the evaluation
    :param variable1: str, name of the column in the df containing the accuracy values of the first group
    :param variable2: str, name of the column in the df containing the accuracy values of the second group
    """
    t_statistic, p_value = ttest_rel(ttest_df[variable1], ttest_df[variable2])

    # Print the results
    print(f"\nT-statistic: {round(t_statistic, 2)}")
    print(f"P-value: {round(p_value, 2)}")

    # Interpret the results
    alpha = 0.05
    if p_value < alpha:
        print("There is a significant difference between the two conditions.")
    else:
        print("There is no significant difference between the two conditions.")

def plot_bar_chart(df):
    """
    Plot bar chart for task times data per participant and version.
    :param df: imported task times DataFrame
    """
    id = df['Participant'].unique()
    versions = df['Version'].unique()
    index = np.arange(len(id))

    bar_width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['#1D2F6F', '#8390FA']

    # Create bars for each version
    for i, version in enumerate(versions):
        version_data = df[df['Version'] == version]
        task_lengths = version_data['Task Length Decimal']
        ax.bar(index + i * bar_width, task_lengths, bar_width, label=version, color=colors[i])

    # Adding labels, title, and legend
    ax.set_xlabel('Participant')
    ax.set_ylabel('Task Length (minutes)')
    ax.set_title('Task Length by Participant and Task Version')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(id)
    ax.legend(title='Version')

    plt.show()


def evaluate_task_times(df):
    """
    Statistical analysis of the task times.
    :param df: DataFrame, imported task times df
    """
    print(f"\n")
    print("#### EVALUATE TASK TIMES: KEYS VS. SPEECH ####")

    # Prepare the df for evaluation
    task_times = prep_task_time_ttest_df(df)

    # Print stats and completed task ratios
    print_stats_runtime(df[df['Version'] == 'Keys'], 'Task Length Decimal', 'keys version', 'mins')
    completed_ratio(df, 'Keys')
    print_stats_runtime(df[df['Version'] == 'Speech'], 'Task Length Decimal', 'speech version', 'mins')
    completed_ratio(df, 'Speech')

    # Run t-test
    run_paired_ttest(task_times, 'Keys', 'Speech')

    # Plot
    plot_bar_chart(df)

def evaluate_runtime_metrics(df):
    """
    Statistical analysis of the query runtimes.
    :param df: DataFrame, imported evaluation df
    """
    info = "#### EVALUATE QUERY TIMES: KEYS VS. SPEECH"
    # Preprocess data
    runtime_df = preprocess_data(df)

    # Create df grouped by task and version with mean and std
    grouped_df = create_evaluation_df(runtime_df, verbose=False)

    print(f"\n")
    print(f"{info} - OBJECT RECOGNITION ####")
    # Prepare object task t-test data
    object_df = prep_version_ttest_df(grouped_df, 'object')

    # Print stats objects
    print_stats_runtime(grouped_df[grouped_df['task'] == 'object'], 'mean', 'object recognition task', 's')

    # Run t-test - object
    run_paired_ttest(object_df, 'keys', 'speech')

    print(f"\n")
    print(f"{info} - COLOR RECOGNITION ####")
    # Prepare color task t-test data
    color_df = prep_version_ttest_df(grouped_df, 'color')

    # Print stats colors
    print_stats_runtime(grouped_df[grouped_df['task'] == 'color'], 'mean', 'color recognition task', 's')

    # Run t-test - color
    run_paired_ttest(color_df, 'keys', 'speech')


def main():
    # Load the task times data
    task_times = import_data(r"utils/MMUI_task_times.csv")

    # Evaluate task times
    evaluate_task_times(task_times)

    # Load the evaluation data
    evaluation_df = import_data(r"utils/main_evaluation_accuracy.csv")

    # Evaluate query times
    evaluate_runtime_metrics(evaluation_df)

if __name__ == "__main__":
    main()