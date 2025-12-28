import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import lightgbm as lgb
from rdkit import Chem
from rdkit.Chem import MACCSkeys, Descriptors,AllChem
import seaborn as sns
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score, GroupKFold, cross_val_predict
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression, Lasso
from skopt.space import Real, Integer, Categorical
from skopt import BayesSearchCV
from scipy.stats import pearsonr, spearmanr

red = '#DD706E'
grey = '#515265'
yellow = '#FAAF3A'
blue = '#3A93C2'
green = '#4CAF50'

# ==================== 1. 数据加载 ====================
train_file_path = "data/train_dataset_fold_5_clean.csv"
test_file_path = "data/test_dataset_fold_5_clean.csv"

train_df = pd.read_csv(train_file_path)
test_df = pd.read_csv(test_file_path)

# ==================== 2. 分别生成溶质和溶剂的描述符 ====================
print("\n 生成溶质和溶剂的分子特征...")


def generate_features_for_smiles(smiles_list, compound_type="compound"):
    """
    为SMILES列表生成分子特征
    compound_type: "solute" 或 "solvent"，用于特征命名
    """
    maccs_list = []
    descriptors_list = []
    extra_list = []
    descriptor_names = [desc[0] for desc in Descriptors._descList]
    descriptor_functions = [desc[1] for desc in Descriptors._descList]

    for idx, smile in enumerate(smiles_list):
        if idx % 100 == 0:
            print(f"  处理第 {idx}/{len(smiles_list)} 个{compound_type}...")

        try:
            mol = Chem.MolFromSmiles(smile)
            if mol is None:
                raise ValueError(f"无效的SMILES: {smile}")
            mol_3d = Chem.AddHs(mol)
            try:
                AllChem.EmbedMolecule(mol_3d)
                embed_success = True
            except:
                embed_success = False
            fingerprint = MACCSkeys.GenMACCSKeys(mol)
            maccs_list.append(list(fingerprint)[1:])
            descriptor_values = [func(mol) for func in descriptor_functions]
            descriptors_list.append(descriptor_values)
            if embed_success:
                try:
                    volume = AllChem.ComputeMolVolume(mol_3d)
                except:
                    volume = np.nan

                try:
                    AllChem.ComputeGasteigerCharges(mol_3d)
                    dipole = 0
                    for i in range(mol_3d.GetNumAtoms()):
                        charge = mol_3d.GetAtomWithIdx(i).GetProp('_GasteigerCharge')
                        if charge != '':
                            pos = mol_3d.GetConformer().GetAtomPosition(i)
                            dipole += float(charge) * pos.x
                except:
                    dipole = np.nan
            else:
                volume = np.nan
                dipole = np.nan

            extra_list.append([volume, dipole])

        except Exception as e:
            print(f"  警告: 处理{compound_type} '{smile[:30]}...' 时出错: {e}")
            maccs_list.append([np.nan] * 166)
            descriptors_list.append([np.nan] * len(descriptor_names))
            extra_list.append([np.nan, np.nan])

    maccs_df = pd.DataFrame(maccs_list, columns=[f'{compound_type}_Maccs_{i}' for i in range(1, 167)])
    descriptors_df = pd.DataFrame(descriptors_list, columns=[f'{compound_type}_{name}' for name in descriptor_names])
    extra_desc_df = pd.DataFrame(extra_list, columns=[f'{compound_type}_Volume', f'{compound_type}_Dipole'])
    return maccs_df, descriptors_df, extra_desc_df


def generate_features_for_both(train_solutes, train_solvents, test_solutes, test_solvents):
    """分别为训练集和测试集的溶质和溶剂生成特征"""
    print("\n为训练集生成特征...")
    print("  生成溶质特征...")
    train_solute_maccs, train_solute_desc, train_solute_extra = generate_features_for_smiles(
        train_solutes, compound_type="solute"
    )

    print("  生成溶剂特征...")
    train_solvent_maccs, train_solvent_desc, train_solvent_extra = generate_features_for_smiles(
        train_solvents, compound_type="solvent"
    )

    print("\n为测试集生成特征...")
    print("  生成溶质特征...")
    test_solute_maccs, test_solute_desc, test_solute_extra = generate_features_for_smiles(
        test_solutes, compound_type="solute"
    )

    print("  生成溶剂特征...")
    test_solvent_maccs, test_solvent_desc, test_solvent_extra = generate_features_for_smiles(
        test_solvents, compound_type="solvent"
    )

    return (train_solute_maccs, train_solute_desc, train_solute_extra,
            train_solvent_maccs, train_solvent_desc, train_solvent_extra,
            test_solute_maccs, test_solute_desc, test_solute_extra,
            test_solvent_maccs, test_solvent_desc, test_solvent_extra)


train_solutes = train_df['Solute_SMILES'].tolist()
train_solvents = train_df['Solvent_SMILES'].tolist()
test_solutes = test_df['Solute_SMILES'].tolist()
test_solvents = test_df['Solvent_SMILES'].tolist()
(train_solute_maccs, train_solute_desc, train_solute_extra,
 train_solvent_maccs, train_solvent_desc, train_solvent_extra,
 test_solute_maccs, test_solute_desc, test_solute_extra,
 test_solvent_maccs, test_solvent_desc, test_solvent_extra) = generate_features_for_both(
    train_solutes, train_solvents, test_solutes, test_solvents
)

# ==================== 3. 特征拼接 ====================
print("\n🔗 拼接溶质和溶剂特征...")


def combine_solute_solvent_features(solute_maccs, solute_desc, solute_extra,
                                    solvent_maccs, solvent_desc, solvent_extra, extra_features):
    """拼接溶质和溶剂的特征"""
    combined = pd.concat([
        solute_maccs.reset_index(drop=True),
        solute_desc.reset_index(drop=True),
        solute_extra.reset_index(drop=True),
        solvent_maccs.reset_index(drop=True),
        solvent_desc.reset_index(drop=True),
        solvent_extra.reset_index(drop=True),
        extra_features.reset_index(drop=True),
    ], axis=1)

    return combined

train_tem = train_df[['temperature/K']].copy()
test_tem = test_df[['temperature/K']].copy()
X_train_combined = combine_solute_solvent_features(
    train_solute_maccs, train_solute_desc, train_solute_extra,
    train_solvent_maccs, train_solvent_desc, train_solvent_extra, train_tem
)
X_test_combined = combine_solute_solvent_features(
    test_solute_maccs, test_solute_desc, test_solute_extra,
    test_solvent_maccs, test_solvent_desc, test_solvent_extra, test_tem
)
y_train = train_df['LogS(mol/L)'].values
y_test = test_df['LogS(mol/L)'].values

# ====== 处理训练集 ======
train_has_nan = np.isnan(X_train_combined).any(axis=1)
train_keep = ~train_has_nan
X_train_combined = X_train_combined[train_keep]
y_train = y_train[train_keep]
train_df = train_df[train_keep].reset_index(drop=True)


# ====== 处理测试集 ======
test_has_nan = np.isnan(X_test_combined).any(axis=1)
test_keep = ~test_has_nan
X_test_combined = X_test_combined[test_keep]
y_test = y_test[test_keep]
test_df = test_df[test_keep].reset_index(drop=True)

print(f"训练集特征形状: {X_train_combined.shape}")
print(f"测试集特征形状: {X_test_combined.shape}")

# ==================== 5. 特征预处理 ====================
print("\n 特征预处理...")


def preprocess_features(X_train, X_test):
    """预处理特征：移除零方差、高相关性和标准化"""
    variance_selector = VarianceThreshold(threshold=0.0)
    X_train_processed = variance_selector.fit_transform(X_train)
    X_test_processed = variance_selector.transform(X_test)
    selected_features = X_train.columns[variance_selector.get_support()]
    X_train_processed = pd.DataFrame(X_train_processed, columns=selected_features)
    X_test_processed = pd.DataFrame(X_test_processed, columns=selected_features)

    print(f"  移除零方差特征后: {X_train_processed.shape[1]} 个特征")
    def remove_high_correlation(df, threshold=0.95):
        corr_matrix = df.corr().abs()
        columns_to_drop = set()

        for col in df.columns:
            if col not in columns_to_drop:
                high_corr_cols = corr_matrix.loc[col, corr_matrix.loc[col, :] > threshold].index.tolist()
                if col in high_corr_cols:
                    high_corr_cols.remove(col)
                columns_to_drop.update(high_corr_cols)

        return df.drop(columns=list(columns_to_drop))
    X_train_processed = remove_high_correlation(X_train_processed, threshold=0.95)
    X_test_processed = X_test_processed[X_train_processed.columns]
    print(f"  移除高相关性特征后: {X_train_processed.shape[1]} 个特征")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_processed)
    X_test_scaled = scaler.transform(X_test_processed)

    return (X_train_scaled, X_test_scaled,
            pd.DataFrame(X_train_scaled, columns=X_train_processed.columns),
            pd.DataFrame(X_test_scaled, columns=X_test_processed.columns),
            scaler)

X_train_scaled, X_test_scaled, X_train_scaled_df, X_test_scaled_df, feature_scaler = preprocess_features(
    X_train_combined, X_test_combined
)


# ==================== 6. PCA降维 ====================
print("\n 执行PCA降维分析...")


def perform_pca_and_export(X_train_scaled, X_test_scaled, X_train_df, X_test_df,
                           variance_threshold=0.95, output_prefix=""):
    """执行PCA分析并输出降维后的描述符"""
    identifier_cols = ['Solute_SMILES', 'Solvent_SMILES']
    if all(col in X_train_df.columns for col in identifier_cols):
        train_identifiers = X_train_df[identifier_cols].copy()
        test_identifiers = X_test_df[identifier_cols].copy()
        X_train_features = X_train_df.drop(identifier_cols, axis=1).values
        X_test_features = X_test_df.drop(identifier_cols, axis=1).values
        feature_names = X_train_df.drop(identifier_cols, axis=1).columns.tolist()
    else:
        train_identifiers = None
        test_identifiers = None
        X_train_features = X_train_scaled
        X_test_features = X_test_scaled
        feature_names = [f'Feature_{i}' for i in range(X_train_scaled.shape[1])]
    pca = PCA()
    X_train_pca_full = pca.fit_transform(X_train_features)
    X_test_pca_full = pca.transform(X_test_features)
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_explained_variance = np.cumsum(explained_variance_ratio)
    pc_number_threshold = np.argmax(cumulative_explained_variance >= variance_threshold) + 1
    pca_summary = pd.DataFrame({
        'Component': range(1, len(explained_variance_ratio) + 1),
        'Variance_Explained': explained_variance_ratio * 100,
        'Cumulative_Variance': cumulative_explained_variance * 100
    })

    print(f"  PCA分析结果:")
    print(f"    总主成分数: {len(explained_variance_ratio)}")
    print(f"    保留 {variance_threshold * 100}% 方差所需主成分: {pc_number_threshold}")
    print(
        f"    前{pc_number_threshold}个主成分解释方差: {pca_summary['Cumulative_Variance'].iloc[pc_number_threshold - 1]:.2f}%")

    X_train_pca_reduced = X_train_pca_full[:, :pc_number_threshold]
    X_test_pca_reduced = X_test_pca_full[:, :pc_number_threshold]
    pca_column_names = [f'PC_{i + 1}' for i in range(pc_number_threshold)]
    if train_identifiers is not None:
        X_train_pca_df = pd.concat([
            train_identifiers.reset_index(drop=True),
            pd.DataFrame(X_train_pca_reduced, columns=pca_column_names)
        ], axis=1)

        X_test_pca_df = pd.concat([
            test_identifiers.reset_index(drop=True),
            pd.DataFrame(X_test_pca_reduced, columns=pca_column_names)
        ], axis=1)
    else:
        X_train_pca_df = pd.DataFrame(X_train_pca_reduced, columns=pca_column_names)
        X_test_pca_df = pd.DataFrame(X_test_pca_reduced, columns=pca_column_names)


    print(f"     降维后的描述符已保存到:")
    print(f"     {output_prefix}pca_reduced_descriptors_train.xlsx")
    print(f"     {output_prefix}pca_reduced_descriptors_test.xlsx")


    if train_identifiers is not None:
        X_train_features_df = pd.concat([
            train_identifiers.reset_index(drop=True),
            pd.DataFrame(X_train_features, columns=feature_names)
        ], axis=1)

        X_test_features_df = pd.concat([
            test_identifiers.reset_index(drop=True),
            pd.DataFrame(X_test_features, columns=feature_names)
        ], axis=1)

        X_train_features_df.to_excel(f'{output_prefix}processed_descriptors_train.xlsx', index=False)
        X_test_features_df.to_excel(f'{output_prefix}processed_descriptors_test.xlsx', index=False)

    return X_train_pca_reduced, X_test_pca_reduced, pca, pca_summary, X_train_pca_df, X_test_pca_df


# 执行PCA并输出描述符
X_train_pca, X_test_pca, pca_model, pca_summary, X_train_pca_df, X_test_pca_df = perform_pca_and_export(
    X_train_scaled, X_test_scaled, X_train_scaled_df, X_test_scaled_df,
    variance_threshold=0.95, output_prefix="solute_solvent_temp_"
)

# ==================== 7. 特征重要性分析 ====================


# ==================== 8. 模型训练和评估 ====================
print("\n 开始模型训练和评估...")

X_train = X_train_pca
X_test = X_test_pca
Y_train = y_train
Y_test = y_test
train_pairs = train_df['Solute_SMILES'] + '_' + train_df['Solvent_SMILES']
G_train = pd.util.hash_pandas_object(train_pairs).values  # 生成唯一 group ID

search_spaces = {
    "DT": {
        "max_depth": Integer(3, 20),
        "splitter": Categorical(['best', 'random']),
        "min_samples_split": Real(0.01, 0.1),
        "min_samples_leaf": Integer(1, 20),
        "max_features": Categorical([ 'sqrt', 'log2']),
    },
    "RF": {
        "n_estimators": Integer(10, 400),
        "max_depth": Integer(3, 20),
        "min_samples_split": Real(0.01, 0.1),
        "min_samples_leaf": Integer(1, 20),
        "max_features": Categorical([ 'sqrt', 'log2']),
        "bootstrap": Categorical([True, False]),
    },
    "NN": {
        "hidden_layer_sizes": Integer(2, 64),
        "alpha": Real(0.0001, 0.1, prior="log-uniform"),
        "learning_rate_init": Real(0.001, 0.1, prior="log-uniform"),
        "activation": Categorical(['relu', 'tanh', 'logistic']),
    },

    "LightGBM": {
        "num_leaves": Integer(10, 400),
        "max_depth": Integer(3, 20),
        "learning_rate": Real(0.01, 0.3, prior="log-uniform"),
        "n_estimators": Integer(100, 1000),
        "bagging_fraction": Real(0.5, 1),
        "feature_fraction": Real(0.5, 1),
        "min_child_samples": Integer(5, 100),
    },
    "MLR": {
        'fit_intercept': Categorical([True, False]),
        'positive': Categorical([True, False])
    },
    "Lasso": {
        "alpha": Real(0.0001, 1, prior="log-uniform"),
        "selection": Categorical(['cyclic', 'random']),
    },
    "kNN": {'n_neighbors': Integer(2, 50),
            'weights': Categorical(["uniform", 'distance']),
            'algorithm': Categorical(['auto', 'ball_tree', 'kd_tree', 'brute']),
            'leaf_size': Integer(10, 100),
            'p': Integer(1, 2),
            },
    "PLS": {'n_components': Integer(2, 6),
            'max_iter': Integer(250, 1000)
            }

}

models = {
    "DT": DecisionTreeRegressor(random_state=0),
    "RF": RandomForestRegressor(random_state=0, n_jobs=6),
    "NN": MLPRegressor(random_state=0),
    "LightGBM": lgb.LGBMRegressor(random_state=0, n_jobs=6),
    "MLR": LinearRegression(),
    "Lasso": Lasso(),
    "PLS": PLSRegression(),
    "kNN": KNeighborsRegressor(n_jobs=6)}


def perform_hp_screening(model_name, X_train, Y_train, G_train, n_iter):
    model = models[model_name]
    search_space = search_spaces[model_name]

    cv = GroupKFold(n_splits=10)

    bscv = BayesSearchCV(
        estimator=model,
        search_spaces=search_space,
        scoring='neg_mean_absolute_error',
        cv=cv,
        n_iter=n_iter,
        n_jobs= 2,
        verbose=0,
        random_state=0
    )

    bscv.fit(X_train, Y_train, groups=G_train)

    return bscv.best_estimator_, bscv.best_params_, bscv.best_score_

model_names = ["DT", "RF",  "NN", "LightGBM", "MLR", "Lasso", "PLS", "kNN"]
results = {}

n_iter = 100

for model_name in model_names:
    search_space = search_spaces[model_name]
    best_model, best_params, best_score = perform_hp_screening(model_name, X_train, Y_train, G_train, n_iter)

    print(model_name, ':  ', round(best_score, 3))

    results[model_name] = {
        'best_estimator': best_model,
        'best_params': best_params,
        'best_score': best_score
    }


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


metrics = ['MAE', 'MedAE', 'RMSE', 'MSE', 'PCC', 'SCC']
train_summary = pd.DataFrame(index=metrics, columns=model_names)
test_summary = pd.DataFrame(index=metrics, columns=model_names)
predictions = pd.DataFrame(index=range(len(Y_test)), columns=model_names)


for model_name in model_names:
    model = results[model_name]['best_estimator']

    cv = GroupKFold(n_splits=10)
    Y_pred_train = cross_val_predict(model, X_train, Y_train, cv=cv, groups=G_train, n_jobs=6)

    Y_train = np.ravel(Y_train)
    Y_pred_train = np.ravel(Y_pred_train)

    train_summary.loc['MAE', model_name] = mean_absolute_error(Y_train, Y_pred_train)
    train_summary.loc['MedAE', model_name] = np.median(np.abs(Y_train - Y_pred_train))
    train_summary.loc['RMSE', model_name] = rmse(Y_train, Y_pred_train)
    train_summary.loc['MSE', model_name] = mean_squared_error(Y_train, Y_pred_train)
    train_summary.loc['PCC', model_name] = pearsonr(Y_train, Y_pred_train)[0]
    train_summary.loc['SCC', model_name] = spearmanr(Y_train, Y_pred_train)[0]
    model.fit(X_train, Y_train)
    Y_pred_test = model.predict(X_test)

    Y_pred_test = np.ravel(Y_pred_test)
    predictions[model_name] = Y_pred_test

    test_summary.loc['MAE', model_name] = mean_absolute_error(Y_test, Y_pred_test)
    test_summary.loc['MedAE', model_name] = np.median(np.abs(Y_test - Y_pred_test))
    test_summary.loc['RMSE', model_name] = rmse(Y_test, Y_pred_test)
    test_summary.loc['MSE', model_name] = mean_squared_error(Y_test, Y_pred_test)
    test_summary.loc['PCC', model_name] = pearsonr(Y_test, Y_pred_test)[0]
    test_summary.loc['SCC', model_name] = spearmanr(Y_test, Y_pred_test)[0]

test_AE = predictions.apply(lambda col: np.abs(col - Y_test), axis=0)

plt.figure(figsize=(10, 4))
plt.rcParams.update({'font.size': 12})

palette = [red] + [blue] * (len(test_AE.columns) - 1)

sns.boxplot(data=test_AE, palette=palette, showfliers=False, showmeans=True, linewidth=1.0,
            meanprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "white", "markersize": 7,
                       "markeredgewidth": 0.5, "markeredgecolor": "black"})

plt.title('')
plt.ylabel('Model Absolute Error')

plt.annotate('a)', xy=(0, 1.06), xycoords="axes fraction", va="top", ha="left", fontsize=12)

prediction_results = pd.DataFrame({
    'Solute_SMILES': test_df['Solute_SMILES'].values,
    'Solvent_SMILES': test_df['Solvent_SMILES'].values,
    'Temperature_K': test_df['temperature/K'].values,
    'True_LogS': Y_test
})

# 添加每个模型的预测
for model_name in model_names:
    prediction_results[f'Pred_{model_name}'] = predictions[model_name]

results_summary = {}
for model_name, res in results.items():
    results_summary[model_name] = {
        'best_score': float(res['best_score']),
        'best_params': res['best_params']
    }


summary_df = pd.DataFrame.from_dict(
    {model: {**res['best_params'], 'best_score': res['best_score']}
     for model, res in results_summary.items()},
    orient='index'
)

summary_df.to_csv("hyperparameter_tuning_results.csv")


prediction_results.to_csv("test_set_predictions_5.csv", index=False)
plt.savefig('Figure_SI_PCA_Model_test_boxplots_5.png', transparent=True, dpi=300)

