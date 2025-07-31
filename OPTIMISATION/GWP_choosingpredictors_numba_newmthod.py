
import numpy as np
import GWP_parameterisation

# Choosing best predictors.

# Choose 5 best predictors for sensitivity (based on p-value or rmse) (heavily depends on outliers, so remove them)
# Choose 5 best predictors for delay (based on p-value or rmse) (heavily depends on outliers, so remove them)
# Do all possible subsets of S and tau predictors. so that is 2^5*2^5 which is 1024 combinations (1000 seconds is like 15 minutes).
# Then for each cardinality, plot the spread of rmse. 
# From that we can find the elbow, then the best set with that cardinality. Or other way around.
# Choose the best set in each cardinality, see where the elbow is.
# Show that the choice of best set is still robust even if 10 predictors are used
#
# how to deal with outlier situation: Dont use them for prediction, bc we know they are bad and we have our reasons.
# The final model will still behave badly for them, so we either ignore them of highlight them with a different colour.


from tqdm import tqdm
import itertools
import pandas as pd
import numpy as np
from sklearn.metrics import root_mean_squared_error
import warnings; warnings.filterwarnings("ignore")
from sklearn.linear_model import LinearRegression
import GWP_parameterisation
from sklearn.feature_selection import r_regression


# List all numerical columns of params

params=pd.read_csv("New_bigtable/Universalparams.csv")
params.set_index('glacier_name',inplace=True)
leave_out=['Langenferner','Kennikot']
# leave_out=['Kennikot']
params.drop(index=leave_out,inplace=True)
names=params.index

station_data_cache = {
  name: pd.read_csv(f"Universalhourlydatasets/{name}.csv")
  for name in names
}

numeric_columns=params.select_dtypes(include="number").columns.values
numeric_columns = [col for col in numeric_columns if not col.lower().startswith("unnamed") and not np.isin(col,['rel_heightfromglacier','Heightfromglacier','CW2','mean_wind_era5','mean_wind_obs','mean_temp_obs','rel_distance_from_terminus','Distance from terminus',"mean_along_valley%",'score','lat','lon','A','B','sensitivity','delay','Down_glacier_direction','Z_max','Z_min','mean_wind_era','localtime',
                                                                                                                	'channel_width_100m_from_lowest_point','rel_heightfromglacier',	'score'	,'mean_along_valley%',	'data_length_days',	'mean_abs_along_valley%','Mean_elev'])]
print(len(numeric_columns))
params.replace(np.nan,0,inplace=True)

# Fit each of them against mean wind speed and find the p_value
def all_subsets(input_list):
    subsets = []
    for r in range(1,8+1):
        subsets.extend(itertools.combinations(input_list, r))
    return subsets


def longlist():
    long_listed_predictors=pd.DataFrame(columns=['bic','pvalue'],index=numeric_columns)
    return long_listed_predictors

# Calculate bic for all subsets in this shortlist

def brute_force(predictors=None,model_coefficient='mean_wind_obs'):
    subsets=all_subsets(predictors)
    bic_df=pd.DataFrame(columns=['bic','subset','N_p','rmse'])

    for i,column in enumerate(tqdm(subsets)):
        # leave=[]
        leave=[]
        # print(bic)
        # try: # leave one out analysis for all stations and gives overall rmse of multiparameter fit
        X = params[list(column)].values
        y = params['mean_wind_obs'].values  # Directly use target column

        rmse = GWP_parameterisation.Validate_U_numba(X, y, len(params))

        # rmse=GWP_parameterisation.Validate_U(leave_out=[],
        #             u_predictors=list(column),
        #            )

        # except:
        #     print("GWP_parameteristaion error, replacing with np.nan")
        #     rmse=np.nan
        # bic_df.loc[i,'bic']=bic
        bic_df.loc[i,'rmse']=rmse
        bic_df.loc[i,'subset']=column
        bic_df.loc[i,'N_p']=len(column)
    bic_df.to_csv("bic_"+model_coefficient+"6.csv")

    print("#### bic ####")
    print(bic_df.sort_values(by='rmse').head(10))
    bic_df.sort_values(by='rmse',inplace=True)
    best_predictor_set=list(bic_df['subset'].iloc[0])
    print("Best Predictor set for",model_coefficient,":",best_predictor_set)

    return bic_df,best_predictor_set


def brute_force_SD(S_predictors=None,D_predictors=None):
    S_subsets=all_subsets(S_predictors)
    D_subsets=all_subsets(D_predictors)
    bic_df=pd.DataFrame(columns=['S_subset','D_subset','rmse','N_p'])
    k=0
    for j,D_column in enumerate(tqdm(D_subsets)):
        for i,S_column in enumerate(tqdm(S_subsets)):
            # leave=[]
            leave=[]
            try:
                rmse=GWP_parameterisation.Validate_U_bar(leave_out=[],
                    S_predictors=list(S_column),
                    D_predictors=list(D_column),
                    plot=False)
            except:
                print("GWP_parameteristaion error, replacing with np.nan")
                rmse=np.nan
            bic_df.loc[k,'rmse']=rmse
            bic_df.loc[k,'S_subset']=str(S_column)
            bic_df.loc[k,'D_subset']=str(D_column)
            bic_df.loc[k,'N_p_S']=len(S_column)
            bic_df.loc[k,'N_p_D']=len(D_column)
            k=k+1
    bic_df.to_csv("bic_SD.csv")

    print("#### bic ####")
    print(bic_df.sort_values(by='rmse').head(10))
    bic_df.sort_values(by='rmse',inplace=True)
    best_S_predictor_set=(bic_df['S_subset'].iloc[0])
    print("Best Predictor set for sensitivity",":",best_S_predictor_set)
    best_D_predictor_set=(bic_df['D_subset'].iloc[0])
    print("Best Predictor set for delay",":",best_D_predictor_set)

    # GWP_parameterisation.Validate_U_bar(leave_out=[],
    #             S_predictors=best_S_predictor_set,
    #             D_predictors=best_D_predictor_set,
    #             plot=True
    # )

    return bic_df,best_S_predictor_set,best_D_predictor_set

def brute_force_SD_numba(S_predictors=None, D_predictors=None):
    """
    Original structure with optimized validation core
    """
    # Precompute all numpy arrays once at the start
    X_S = params[S_predictors].values.astype(np.float64)
    X_D = params[D_predictors].values.astype(np.float64)
    y_S = params['sensitivity'].values.astype(np.float64)
    y_D = params['delay'].values.astype(np.float64)
    
    # Precompute temporal data
    temp_models = np.array([station_data_cache[name]['temp_model'].values for name in names])
    ws_obs = np.array([station_data_cache[name]['ws'].values for name in names])
    
    # Get column indices for the predictors
    S_col_indices = {col: idx for idx, col in enumerate(S_predictors)}
    D_col_indices = {col: idx for idx, col in enumerate(D_predictors)}
    
    S_subsets = all_subsets(S_predictors)
    D_subsets = all_subsets(D_predictors)
    
    bic_df = pd.DataFrame(columns=['S_subset', 'D_subset', 'rmse', 'N_p_S', 'N_p_D'])
    k = 0
    
    for D_column in tqdm(D_subsets, desc="D subsets"):
        # Convert to column indices
        D_idx = np.array([D_col_indices[col] for col in D_column], dtype=np.int64)
        
        for S_column in tqdm(S_subsets, desc="S subsets", leave=False):
            # Convert to column indices
            S_idx = np.array([S_col_indices[col] for col in S_column], dtype=np.int64)
            
            try:
                # Use the optimized validation
                rmse = GWP_parameterisation.validate_U_bar_numba(
                    S_idx, D_idx, 
                    X_S, X_D, y_S, y_D,
                    temp_models, ws_obs
                )
            except Exception as e:
                print(f"Validation error: {e}, replacing with np.nan")
                rmse = np.nan
            
            # Store results in original format
            bic_df.loc[k] = {
                'S_subset': str(S_column),
                'D_subset': str(D_column),
                'rmse': rmse,
                'N_p_S': len(S_column),
                'N_p_D': len(D_column)
            }
            k += 1

            
    
    # Save and report results (unchanged from original)
    bic_df.to_csv("bic_SD.csv")
    print("#### bic ####")
    print(bic_df.sort_values(by='rmse').head(10))
    
    bic_df.sort_values(by='rmse', inplace=True)
    best_S = eval(bic_df['S_subset'].iloc[0])  # Convert string back to tuple
    best_D = eval(bic_df['D_subset'].iloc[0])
    
    print("Best Predictor set for sensitivity:", best_S)
    print("Best Predictor set for delay:", best_D)
    bic_df.to_csv("bic_SD.csv")
    return bic_df, best_S, best_D


from joblib import Parallel, delayed
from tqdm import tqdm

def brute_force_SD_parallel(S_predictors=None, D_predictors=None, n_jobs=12):
    """
    Parallel version maintaining original format
    """
    # Precompute all numpy arrays once at the start
    X_S = params[S_predictors].values.astype(np.float64)
    X_D = params[D_predictors].values.astype(np.float64)
    y_S = params['sensitivity'].values.astype(np.float64)
    y_D = params['delay'].values.astype(np.float64)
    
    # Precompute temporal data
    temp_models = np.array([station_data_cache[name]['temp_model'].values for name in names])
    ws_obs = np.array([station_data_cache[name]['ws'].values for name in names])
    
    # Create index mappings
    S_col_indices = {col: idx for idx, col in enumerate(S_predictors)}
    D_col_indices = {col: idx for idx, col in enumerate(D_predictors)}
    
    # Generate all combinations upfront
    S_subsets = all_subsets(S_predictors)
    D_subsets = all_subsets(D_predictors)
    tasks = [(s, d) for d in D_subsets for s in S_subsets]  # Note reversed order for progress
    
    # Parallel processing with progress bar
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_sd_pair)(
            s, d, S_col_indices, D_col_indices,
            X_S, X_D, y_S, y_D, temp_models, ws_obs
        ) for s, d in tqdm(tasks, desc="Processing SD pairs")
    )
    
    # Convert results to DataFrame
    bic_df = pd.DataFrame(results)
    bic_df.sort_values(by='rmse', inplace=True)
    
    # Post-processing identical to original
    print("#### bic ####")
    print(bic_df.head(10))
    
    best_S = eval(bic_df['S_subset'].iloc[0])
    best_D = eval(bic_df['D_subset'].iloc[0])
    bic_df.to_csv("bic_SD.csv")
    return bic_df, best_S, best_D

from joblib import Parallel, delayed
from tqdm.auto import tqdm
import math

def brute_force_SD_batched(S_predictors=None, D_predictors=None, 
                          n_jobs=12, batch_size=1):
    """
    Batched parallel processing with same output format
    """
    # Precompute data and generate tasks (same as before)
    X_S = params[S_predictors].values.astype(np.float64)
    X_D = params[D_predictors].values.astype(np.float64)
    y_S = params['sensitivity'].values.astype(np.float64)
    y_D = params['delay'].values.astype(np.float64)
    
    temp_models = np.array([station_data_cache[name]['temp_model'].values for name in names])
    ws_obs = np.array([station_data_cache[name]['ws'].values for name in names])
    
    S_col_indices = {col: idx for idx, col in enumerate(S_predictors)}
    D_col_indices = {col: idx for idx, col in enumerate(D_predictors)}
    
    S_subsets = all_subsets(S_predictors)
    D_subsets = all_subsets(D_predictors)
    tasks = [(s, d) for d in D_subsets for s in S_subsets]

    # Split tasks into batches
    n_batches = math.ceil(len(tasks) / batch_size)
    all_results = []

    # Process batches with separate progress bars
    with tqdm(total=len(tasks), desc="Total progress") as pbar:
        for batch_idx in range(n_batches):
            start = batch_idx * batch_size
            end = min((batch_idx + 1) * batch_size, len(tasks))
            batch_tasks = tasks[start:end]

            # Process one batch
            batch_results = Parallel(n_jobs=n_jobs)(
                delayed(process_sd_pair)(
                    s, d, S_col_indices, D_col_indices,
                    X_S, X_D, y_S, y_D, temp_models, ws_obs
                ) for s, d in tqdm(batch_tasks, 
                                 desc=f"Batch {batch_idx+1}/{n_batches}",
                                 leave=False)
            )

            all_results.extend(batch_results)
            pbar.update(len(batch_tasks))

    # Convert to DataFrame (same as before)
    bic_df = pd.DataFrame(all_results)
    bic_df.sort_values(by='rmse', inplace=True)
    
    print("#### bic ####")
    print(bic_df.head(10))
    
    best_S = eval(bic_df['S_subset'].iloc[0])
    best_D = eval(bic_df['D_subset'].iloc[0])
    bic_df.to_csv("bic_SD.csv")
    return bic_df, best_S, best_D


def brute_force_batched(predictors=None, n_jobs=12, batch_size=1000,model_coefficient='mean_wind_obs'):
    """
    Batched parallel processing with same output format
    """
    # Precompute data and generate tasks (same as before)
    X_p = params[predictors].values.astype(np.float64)
    y_p = params[model_coefficient].values.astype(np.float64)
    
    temp_models = np.array([station_data_cache[name]['temp_model'].values for name in names])
    ws_obs = np.array(params[model_coefficient].values.astype(np.float64))
    
    P_col_indices = {col: idx for idx, col in enumerate(predictors)}

    P_subsets = all_subsets(predictors)
    tasks = [(p) for p in P_subsets]

    # Split tasks into batches
    n_batches = math.ceil(len(tasks) / batch_size)
    all_results = []

    # Process batches with separate progress bars
    with tqdm(total=len(tasks), desc="Total progress") as pbar:
        for batch_idx in range(n_batches):
            start = batch_idx * batch_size
            end = min((batch_idx + 1) * batch_size, len(tasks))
            batch_tasks = tasks[start:end]

            # Process one batch
            batch_results = Parallel(n_jobs=n_jobs)(
                delayed(process_u)(
                    p, P_col_indices,
                    X_p, y_p, ws_obs
                ) for p in tqdm(batch_tasks, 
                                 desc=f"Batch {batch_idx+1}/{n_batches}",
                                 leave=False)
            )

            all_results.extend(batch_results)
            pbar.update(len(batch_tasks))

    # Convert to DataFrame (same as before)
    bic_df = pd.DataFrame(all_results)
    bic_df.sort_values(by='rmse', inplace=True)
    
    print("#### bic ####")
    print(bic_df.head(10))
    
    best_p = eval(bic_df['subset'].iloc[0])
    bic_df.to_csv("bic_"+model_coefficient+".csv")
    return bic_df, best_p

# Keep process_sd_pair identical to previous implementation

def process_sd_pair(S_column, D_column, S_col_indices, D_col_indices,
                   X_S, X_D, y_S, y_D, temp_models, ws_obs):
    """Process one S/D subset pair"""
    try:
        # Convert to indices
        S_idx = np.array([S_col_indices[col] for col in S_column], dtype=np.int64)
        D_idx = np.array([D_col_indices[col] for col in D_column], dtype=np.int64)
        
        # Calculate RMSE
        rmse = GWP_parameterisation.validate_U_bar_numba(
            S_idx, D_idx, X_S, X_D, y_S, y_D,
            temp_models, ws_obs
        )
    except Exception as e:
        rmse = np.nan
    
    return {
        'S_subset': str(S_column),
        'D_subset': str(D_column),
        'rmse': rmse,
        'N_p_S': len(S_column),
        'N_p_D': len(D_column)
    }

def process_u(P_column, P_col_indices,
                   X_p,y_p,u_obs):
    """Process one S/D subset pair"""
    try:
        # Convert to indices
        P_idx = np.array([P_col_indices[col] for col in P_column], dtype=np.int64)
        # Calculate RMSE
        rmse = GWP_parameterisation.Validate_U_numba_batched(
            P_idx, X_p, y_p, u_obs
        )
    except Exception as e:
        rmse = np.nan
    
    return {
        'subset': str(P_column),
        'rmse': rmse,
        'N_p': len(P_column)
    }
# Short list top n predictors based on coefficient of determination    
# short_listed_predictors=shortlist(n_predictors=len(numeric_columns),model_coefficient='mean_wind_obs')
# print(short_listed_predictors)
    
######## UNCOMMENT FROM HERE

# bic_df,_=brute_force(numeric_columns,model_coefficient='mean_wind_obs')

bic_df,_=brute_force_batched(numeric_columns,model_coefficient='mean_wind_obs',batch_size=1000000)
## Plotting elbow curve of RMSE
import matplotlib.pyplot as plt
import seaborn as sns
# bic_df=pd.read_csv("bic_U_recursive.csv")
sns.boxenplot(data=bic_df,x='N_p',y='rmse')
bic_df.sort_values(by='rmse',inplace=True)
best_predictor_set=(bic_df['subset'].iloc[0])
print("Best Predictor set",":",best_predictor_set)
plt.show()

# plt.figure()
# # plt.scatter(bic_df['bic'],bic_df['rmse'],c=bic_df['N_p'])
# plt.scatter(bic_df['N_p'],bic_df['rmse'],c=bic_df['bic'])
# plt.colorbar()
# plt.show()

# bic_df,_=recursive_elimination_reverse()
# ## Plotting elbow curve of RMSE
# import matplotlib.pyplot as plt
# plt.figure()
# plt.scatter(bic_df['N_p'],bic_df['rmse'])
# plt.show()


# SD predictors


# Convert all data to NumPy arrays upfront

####### UNCOMMENT FROM HERE

# S_short_listed_predictors=numeric_columns
# D_short_listed_predictors=numeric_columns
# print(len(S_short_listed_predictors))
# print(len(D_short_listed_predictors))

# # bic_df,_,_=brute_force_SD_numba(S_predictors=S_short_listed_predictors,D_predictors=D_short_listed_predictors)
# # bic_df,_,_=brute_force_SD_parallel(S_predictors=S_short_listed_predictors,D_predictors=D_short_listed_predictors,n_jobs=12)
# bic_df,_,_=brute_force_SD_batched(S_predictors=S_short_listed_predictors,D_predictors=D_short_listed_predictors,batch_size=1000000)
# # bic_df,_,_=brute_force_SD(S_predictors=S_short_listed_predictors,D_predictors=D_short_listed_predictors)

# bic_df.to_csv("bic_SD.csv")
# print("#### bic ####")
# print(bic_df.sort_values(by='rmse').head(10))
# bic_df.sort_values(by='rmse',inplace=True)
# best_S_predictor_set=(bic_df['S_subset'].iloc[0])
# print("Best Predictor set for sensitivity",":",best_S_predictor_set)
# best_D_predictor_set=(bic_df['D_subset'].iloc[0])
# print("Best Predictor set for delay",":",best_D_predictor_set)
# ## Plotting elbow curve of RMSE
# import matplotlib.pyplot as plt
# import seaborn as sns
# plt.figure()
# plt.title("Sensitivity")
# sns.boxenplot(data=bic_df,x='N_p_S',y='rmse')
# plt.show()
# plt.figure()
# sns.boxenplot(data=bic_df,x='N_p_D',y='rmse',color='red')
# plt.title("Response time")
# plt.show()