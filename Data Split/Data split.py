import pandas as pd
import numpy as np

file_path = 'Output_Step_2_for_Training.xlsx'

data = pd.read_excel(file_path)
data['temperature/K(conver)'] = 1/(1/ data['temperature/K'] * 2 /(1/243.15-1/523) + (243.15+523)/(243.15-523))
data['A(conver)'] = data['A'] + data['B']*(243.15+523)/(2*243.15*523)
data['B(conver)'] = data['B']*(523-243.15)/(2*243.15*523)

group_columns = ['Solute_SMILES', 'Solvent_SMILES']
groups = data[group_columns].drop_duplicates()
num_folds = min(10, len(groups))

fold_indices = np.random.randint(0, num_folds, size=len(groups))
group_to_fold = {tuple(group): fold for group, fold in zip(groups.itertuples(index=False), fold_indices)}

grouped_data = {tuple(group): data[(data[group_columns[0]] == group.iloc[0]) & (data[group_columns[1]] == group.iloc[1])] for _, group in groups.iterrows()}

train_sets = []
test_sets = []

for i in range(num_folds):
    train_data = []
    test_data = []

    for group, group_df in grouped_data.items():
        if group_to_fold[group] == i:
            test_data.append(group_df)
        else:
            train_data.append(group_df)

    train_data = pd.concat(train_data) if train_data else pd.DataFrame()
    test_data = pd.concat(test_data) if test_data else pd.DataFrame()

    train_sets.append(train_data)
    test_sets.append(test_data)

    print(f"Fold {i + 1}:")
    print("len_train_data:", len(train_data))
    print("len_test_data:", len(test_data))

for i, (train_data, test_data) in enumerate(zip(train_sets, test_sets)):
    train_data.to_csv(f'train_dataset_fold_{i + 1}.csv', index=False)
    test_data.to_csv(f'test_dataset_fold_{i + 1}.csv', index=False)

print("All folded datasets have been saved.")
