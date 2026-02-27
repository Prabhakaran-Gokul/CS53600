# congestion_prediction.py
from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, make_scorer
from typing import Tuple
# Add this to congestion_prediction.py
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, train_test_split

from typing import List, Tuple
from cs536.assignment_2 import ASSIGNMENT_2_PATH

def prep_data(
    csv_path: Path = ASSIGNMENT_2_PATH / "results" / "q2_combined.csv",
    alpha: float = 0.1,
    beta: float = 0.1,
    get_all: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    part a) of q3 building the dataset 


    Args:
        csv_path: path to csv file
        one-hot: whether to use one hot encoding or not
        get_all: whether to return all ip addresses or just a time series for one address, by default returns just one (first one)

    Returns:
        x: the features with lag and rolling window
        y: the labels ()

    """
    
    df : pd.DataFrame = pd.read_csv(csv_path)

    ## drop extra rows
    df.drop(columns=["ip", "t_mid"], inplace=True)
    #df.drop(columns=[])

    ## combine total_retrans, lost_pkts, retrans_pkgs into one metric
    df['loss'] = df['total_retrans'] + df['lost_pkts'] + df['retrans_pkts']

    ## remove the other loss metrcs
    df.drop(columns=['total_retrans', 'lost_pkts', 'retrans_pkts'], inplace=True)

    ## define constant for creating time series:
    lags : List[int] = [1,2]
    rolling_window : List[int] = [2,3] ## must be at least 2

    ## output dataframe 
    out : List[pd.DataFrame] = []

    ## create time series 
    for ip, group in df.groupby("destination", sort=False):
        group = group.sort_values('ts').copy() ## group now has the sorted time for an IP

        ## create lags
        for l in lags:
            group[f'lag_{l}'] = group['snd_cwnd'].shift(l)

        ## create windows
        for window in rolling_window:
            group[f"rolling_mean_{window}"] = group['snd_cwnd'].shift(1).rolling(window).mean()
            group[f"rolling_std_{window}"] = group['snd_cwnd'].shift(1).rolling(window).std()
            #print(group['snd_cwnd'].shift(1).rolling(window).std())

        
        ## preparing objective function
        #group["eta_prev"] = (group["goodput_bps"].fillna(0.0) - alpha * group["rtt_us"].fillna(0.0) - beta * group["loss_step"].fillna(0.0))

        #group["series_id"] = ip
        #group['y_target'] = group['snd_cwnd'].shift(-1) ## shift snd_cwnd up so we are precdicting next window 

        ## shift objective
        group["objective"] = group["goodput_bps"].shift(-1) - alpha * group["rtt_us"].shift(-1) - beta * group["loss"].shift(-1)
        group['y_target'] = group["objective"]
        out.append(group)

        if not get_all:
            break



    feat_df = pd.concat(out, axis=0)
    #feat_df = feat_df.dropna(subset=['y_target'] + [c for c in feat_df.columns if c.startswith('lag') or c.startswith('roll')])
    feat_df = feat_df.dropna(subset=['y_target'] + [c for c in feat_df.columns if c.startswith('lag') or c.startswith('roll') or c == 'objective'])

    ## get features     
    x = feat_df.drop(columns=["snd_cwnd", "y_target", "objective"])
    # x = feat_df.drop(columns=["snd_cwnd", "y_target"])

    ## get labels
    y = feat_df['y_target']

    return x, y, feat_df

def get_split(
    x : pd.DataFrame, 
    y : pd.DataFrame,
    feat : pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits into train and test

    Args:
        x: features
        y: labels

    Returns:
        x_train: feature train split 
        x_test: feature test split
        y_train: label train split
        y_test: label test split
        feat_train: train split of everything 
        feat_test: test split of everything  
    """
    
    ## take the last n values of each destination as train test split
    def get_last_n(x : pd.DataFrame, y : pd.DataFrame, feat : pd.DataFrame, n : int = 1) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        x_train = x.head(len(x) - n)
        x_test = x.tail(n)
        y_train = y.head(len(y) - n)
        y_test = y.tail(n)
        feat_train = feat.head(len(feat) - n)
        feat_test = feat.head(n)

        return x_train, x_test, y_train, y_test, feat_train, feat_test

    x_train_list = []
    x_test_list = []
    y_train_list = []
    y_test_list = []
    feat_train_list = []
    feat_test_list = []

    for ip,x_group in x.groupby("destination"):
        #x_train, x_test, y_train, y_test = get_last_n(x[])
        y_group = y.loc[x_group.index]  # align labels to this destination group
        feat_group = feat.loc[x_group.index]
        x_train, x_test, y_train, y_test, feat_train, feat_test= get_last_n(x_group, y_group, feat_group, n=3)
        x_train_list.append(x_train)
        x_test_list.append(x_test)
        y_train_list.append(y_train)
        y_test_list.append(y_test)
        feat_train_list.append(feat_train)
        feat_test_list.append(feat_test)

    x_train = pd.concat(x_train_list)
    x_test = pd.concat(x_test_list)
    y_train = pd.concat(y_train_list)
    y_test = pd.concat(y_test_list)
    feat_train = pd.concat(feat_train_list)
    feat_test = pd.concat(feat_test_list)

    return x_train, x_test, y_train, y_test, feat_train, feat_test




def objective_function(X : pd.DataFrame, y : pd.DataFrame,
                    key_col : str = "destination", ts_col : str = "ts",
                    alpha : float = 0.1, beta : float = 0.1):

    sid = X[key_col].to_numpy()
    valid = sid[1:] == sid[:-1] ## mask of what is valid 

    return (X['goodput_bps'][1:] - alpha * X['rtt_us'][1:] - beta * X['loss'][1:])[valid]

def singe_ip_objective(X, alpha : float = 0.1, beta : float = 0.1):
    return (X['goodput_bps'][1:] - alpha * X['rtt_us'][1:] - beta * X['loss'][1:])


def predict_next_cwnd(
    csv_path:  Path = ASSIGNMENT_2_PATH / "results" / "q2_combined.csv"
):
    """
    Performts periction
    """

    ## alpha and beta
    alpha = 0.1
    beta = 0.1
        
    # 1) Load data
    x, y, feat_df = prep_data(csv_path)

    # 2) get split
    x_train, x_test, y_train, y_test, feat_train, feat_test = get_split(x, y, feat_df)
    x_train.drop(columns=["destination"], inplace=True)
    x_test.drop(columns=["destination"], inplace=True)

    # 3) Feature set (only columns that actually exist)
    feature_cols = x_train.columns

    # 4) get X
    X_train = x_train.copy()

    # 5) Model pipeline
    pre = ColumnTransformer(
        transformers=[("num", Pipeline([
                ('imp', SimpleImputer(strategy = "median")),
                ('sc', StandardScaler())
            ]), feature_cols)],
    )

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
    )

    pipe = Pipeline([
        ("pre", pre),
        ("model", model),
    ])

    # 6) Search
    tscv = TimeSeriesSplit(n_splits=3, gap=2)
    grid = GridSearchCV(
        pipe,
        param_grid={'model__max_depth': [6, 10, None]},
        cv=tscv,
        scoring='neg_mean_absolute_error',
        #scoring=lambda est, X, y: singe_ip_objective(X),
        n_jobs=-1
    )

    # 7) Train
    grid.fit(X_train, y_train)


    ## 8) get possible cwnd values -> possible congestion window changes by looking as past congestion window changes
    cwnd_choices = feat_train['snd_cwnd']

    def max_ojective(row):

        scores = []
        for cwnd in cwnd_choices:
            input = pd.DataFrame([{**row, 'snd_cwnd':cwnd}])
            scores.append((cwnd, grid.predict(input)[0]))
        return max(scores, key= lambda x: x[1])[0]

    counter = 0
    for i, row in x_test.iterrows():

        print(f"Row {counter} predicted cwnd is {max_ojective(row)}. Actual cwnd is {feat_test['snd_cwnd'].iloc[counter]}")
        counter += 1


if __name__ == "__main__":
    predict_next_cwnd()






















'''
def transform_with_pca(
    x: pd.DataFrame,
    n_components: float = 0.95,  # keep enough PCs to explain 95% variance
):
    """
    Runs PCA on numeric TCP features and returns:
      - transformed dataframe with principal components
      - fitted PCA object
      - feature loading matrix

    Honestly if they want a one click run this part won't really be part of it ... 
    """
    

    feature_candidates = x.columns

    X = x.copy()

    # Impute + scale (important for PCA)
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_imp = imputer.fit_transform(X)
    X_scaled = scaler.fit_transform(X_imp)

    # PCA
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    # Build PC dataframe
    pc_cols = [f"PC{i+1}" for i in range(X_pca.shape[1])]
    pca_df = pd.DataFrame(X_pca, columns=pc_cols, index=x.index)

    # Optional: keep identifiers/time if present
    #keep_cols = [c for c in ["ip", "destination", "ts"] if c in x.columns]
    #out_df = pd.concat([x[keep_cols].reset_index(drop=True), pca_df.reset_index(drop=True)], axis=1)

    # Loadings (how original features contribute to each PC)
    loadings = pd.DataFrame(
        pca.components_.T,
        index=X.columns,
        columns=pc_cols,
    )

    print("Explained variance ratio per PC:")
    print(pd.Series(pca.explained_variance_ratio_, index=pc_cols))
    print(f"\nTotal explained variance: {pca.explained_variance_ratio_.sum():.4f}")

    print("\nTop contributing features per PC:")
    for pc in pc_cols:
        top = loadings[pc].abs().sort_values(ascending=False).head(3).index.tolist()
        print(f"{pc}: {top}")

    return pca_df, pca, loadings


'''
