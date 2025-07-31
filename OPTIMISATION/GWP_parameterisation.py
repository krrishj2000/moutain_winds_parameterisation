# Model parameterisation

# Contains model train, model test, and model validate

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import HuberRegressor,LinearRegression
from sklearn.metrics import root_mean_squared_error,r2_score
import matplotlib
import seaborn as sns
from numba import njit

cmap=matplotlib.colormaps['plasma']

params=pd.read_csv(r'New_bigtable/Universalparams.csv')
params.replace(np.nan,0,inplace=True)


params.set_index('glacier_name',inplace=True)
leave_out=['Langenferner','Kennikot']
# leave_out=['Kennikot']
params.drop(index=leave_out,inplace=True)
names=params.index

Lr=LinearRegression()

station_data_cache = {
  name: pd.read_csv(f"Universalhourlydatasets/{name}.csv")
  for name in names
}


def u_parameterisation_train(X_data, y_data):
    return LinearRegression().fit(X_data, y_data)


def S_parameterisation_train(X_data, y_data):
    return LinearRegression().fit(X_data, y_data)

def D_parameterisation_train(X_data, y_data):
    return LinearRegression().fit(X_data, y_data)

    
def Model(u,S,D,name):
    station_data=station_data_cache[name]
    # print('Universalhourlydatasets//'+name+'.csv')
    A=S
    B=-D*S
    linear_model=station_data.copy()
    linear_model['temp_model']=linear_model['temp_model']-linear_model['temp_model'].mean()
    linear_model['ws']=linear_model['ws']-linear_model['ws'].mean()
    linear_model['ws_model']=linear_model['ws_model']-linear_model['ws_model'].mean()

    lres_wind_speed=np.array(((A*linear_model['temp_model']+B*linear_model['temp_model'].diff())).replace(np.nan,0))+u
    return lres_wind_speed[1:]




def Validate_U_bar(leave_out=[],S_predictors=[],D_predictors=[],plot=False):
        
    x_data_sensitivity=params[S_predictors]
    x_data_delay=params[D_predictors]
    y_data_sensitivity=params['sensitivity']
    y_data_delay=params['delay']
    
    u_obs_list=[]
    u_era_list=[]
    u_our_list=[]
    
    
    for name in names:
        if np.isin(name,leave_out):
            continue
        leave=leave_out+[]
        # Training S and D based on all other stations
        S_parameterisation=S_parameterisation_train(X_data=x_data_sensitivity.drop(index=leave),y_data=y_data_sensitivity.drop(index=leave))
        x_validate_sensitivity=x_data_sensitivity.loc[[name]]
        S=S_parameterisation.predict(x_validate_sensitivity)

        D_parameterisation=D_parameterisation_train(X_data=x_data_delay.drop(index=leave),y_data=y_data_delay.drop(index=leave))
        x_validate_delay=x_data_delay.loc[[name]]
        D=D_parameterisation.predict(x_validate_delay)
        # Predicting S and D based on that
        
        # Putting these into model and getting u(t)
        u=0
        # print(name)
        u_bar=Model(u,S,D,name)
        u_our=list(u+u_bar)

        # Compare with observation and era5
        station_data=station_data_cache[name]
        
        station_data['ws']=station_data['ws']-station_data['ws'].mean()
        station_data['ws_model']=station_data['ws_model']-station_data['ws_model'].mean()
        
        u_obs=list(station_data['ws'].values)[1:]
        u_era=list(station_data['ws_model'].values)[1:]

        if len(u_obs)!=len(u_era):
            print(name)
            # break
        
        # Appending this to a list for plotting together
        
        u_obs_list=u_obs_list+u_obs
        u_era_list=u_era_list+u_era
        u_our_list=u_our_list+u_our
    
    winds_df=pd.DataFrame()
    winds_df['u_era']=u_era_list#-winds_df['u_obs']
    winds_df['u_obs']=u_obs_list
    winds_df['u_our']=u_our_list
    winds_df['glacier_number']=np.arange(len(u_our)*(len(names)-len(leave_out)))//len(u_obs)
    
    
    if plot:    
        plt.rcParams["font.family"] = "Helvetica"  
        plt.figure(figsize=(9,8))
        plt.rcParams['font.size']=20
        plt.grid(True,zorder=-1)
        plt.axline((0,0),(1,1),color='black',linewidth=2,linestyle='-')
        import seaborn as sns
        sns.kdeplot(data=winds_df, x='u_obs',y='u_era',ax=plt.gca(),color='lightskyblue',fill=True,levels=20,alpha=0.5,thresh=0.4,linewidth=0.5)
        sns.kdeplot(data=winds_df, x='u_obs',y='u_our',ax=plt.gca(),color='deeppink',fill=True,levels=20,alpha=0.5,thresh=0.3,linewidth=0.5)
        plt.scatter(winds_df['u_obs'],winds_df['u_era'],label="R2_era5= "+str(round(r2_score(winds_df['u_obs'],winds_df['u_era']),3)),edgecolor="black",s=25,linewidth=0.5,color=cmap(winds_df['glacier_number']/34),zorder=2,alpha=0.9)
        plt.scatter(winds_df['u_obs'],winds_df['u_our'],label="R2_ours= "+str(round(r2_score(winds_df['u_obs'],winds_df['u_our']),3)),marker='^',edgecolor="black",s= 25,linewidth=0.5,color=cmap(winds_df['glacier_number']/34),zorder=2,alpha=0.9)
        # plt.axvline()
        plt.xlabel(r"$u'_{obs}\:(\text{ms}{}^{-1})$")
        plt.ylabel(r"$u'_{pred}\:(\text{ms}{}^{-1})$")
        plt.legend(loc='upper left')
        ax = plt.gca()
        ax.set_facecolor('white')  # Light gray background color
        ax.yaxis.set_ticks_position('both')
        ax.xaxis.set_ticks_position('both')
        # Show the plot
        # plt.xlim(-2,2)
        # plt.ylim(2,2)
        plt.minorticks_on()
        plt.tight_layout()
        plt.show()
    
    return root_mean_squared_error(winds_df['u_obs'],winds_df['u_our'])

@njit
def validate_U_bar_numba(S_indices, D_indices, X_S, X_D, y_S, y_D, 
                        all_temp_models, all_ws_obs, delta_t=1):
    """
    Corrected version matching original validation logic
    """
    n_glaciers = X_S.shape[0]
    total_error = 0.0
    valid_glaciers = 0
    
    # Pre-center temperature data (matches Model()'s behavior)
    centered_temps = np.zeros_like(all_temp_models)
    for i in range(n_glaciers):
        temp = all_temp_models[i]
        centered_temps[i] = temp - temp.mean()
    
    # Main validation loop
    for i in range(n_glaciers):
        # Leave-one-out mask
        mask = np.ones(n_glaciers, dtype=np.bool_)
        mask[i] = True
        
        # ===== Train S Model =====
        X_S_train = np.hstack((np.ones((n_glaciers, 1)), X_S[mask][:, S_indices]))
        y_S_train = y_S[mask]
        
        # Solve with intercept (like LinearRegression)
        XtX_S = X_S_train.T @ X_S_train
        Xty_S = X_S_train.T @ y_S_train
        beta_S = np.linalg.solve(XtX_S, Xty_S)
        
        # ===== Train D Model =====
        X_D_train = np.hstack((np.ones((n_glaciers, 1)), X_D[mask][:, D_indices]))
        y_D_train = y_D[mask]
        
        XtX_D = X_D_train.T @ X_D_train
        Xty_D = X_D_train.T @ y_D_train
        beta_D = np.linalg.solve(XtX_D, Xty_D)
        
        # ===== Prediction =====
        # Get centered temp for this glacier
        temp = centered_temps[i]
        
        
        # Predict S and D with intercept
        S_pred = beta_S[0] + np.sum(X_S[i, S_indices] * beta_S[1:])
        D_pred = beta_D[0] + np.sum(X_D[i, D_indices] * beta_D[1:])
        B = -D_pred * S_pred
        
        # ===== Model Simulation =====
        # Match Model()'s calculation
        n_steps = len(temp)
        ws_pred = np.zeros(n_steps)
        for t in range(n_steps):
            if t == 0:
                temp_diff = 0.0  # Matches .diff().fillna(0)
            else:
                temp_diff = temp[t] - temp[t-1]
            
            ws_pred[t] = S_pred * temp[t] + B * temp_diff / delta_t
        
        # Center observed wind speed (matches Model()'s behavior)
        ws_obs = all_ws_obs[i] - all_ws_obs[i].mean()
        ws_pred=ws_pred[1:]
        ws_obs=ws_obs[1:]


        # Accumulate error
        total_error += np.sum((ws_obs - ws_pred)**2)
        valid_glaciers += 1
    
    return np.sqrt(total_error / (valid_glaciers * len(ws_pred)))





# @njit
def Validate_U(u_predictors=np.array([],dtype='int64'),leave_out=[]):  
    x_data_meanwind=params[u_predictors]
    y_data_meanwind=params['mean_wind_obs']
    u_obs_list=[]
    u_era_list=[]
    u_our_list=[]
    for name in names:
        if np.isin(name,leave_out):
            continue
        leave=[]
        # Training u based on all other stations
        u_parameterisation=u_parameterisation_train(X_data=x_data_meanwind.drop(index=leave),y_data=y_data_meanwind.drop(index=leave))
        x_validate_meanwind=x_data_meanwind.loc[[name]]
        u_our=list(u_parameterisation.predict(x_validate_meanwind))
        # Compare with observation and era5
        station_data=station_data_cache[name]
        u_obs=list([station_data['ws'].mean()])
        u_era=list([station_data['ws_model'].mean()])
        
        # Appending this to a list for plotting together
        u_obs_list=u_obs_list+u_obs
        u_era_list=u_era_list+u_era
        u_our_list=u_our_list+u_our
    
    winds_df=pd.DataFrame()
    winds_df['u_era']=u_era_list#-winds_df['u_obs']
    winds_df['u_obs']=u_obs_list
    winds_df['u_our']=u_our_list
    winds_df['glacier_number']=np.arange(len(u_our)*(len(names)))//len(u_obs)
    return root_mean_squared_error(winds_df['u_obs'],winds_df['u_our'])

from numba import njit
import numpy as np


@njit
def Validate_U_numba(X_all, y_true, n_glaciers):
    """
    Optimized Numba version
    Args:
    - X_all: 2D array (n_glaciers, n_features) of predictors
    - y_true: 1D array of true mean wind speeds
    - n_glaciers: Total number of glaciers
    """
    n_features = X_all.shape[1]
    residuals = np.zeros(n_glaciers)
    
    for i in range(n_glaciers):
        mask = np.ones(n_glaciers, dtype=np.bool_)
        mask[i] = True
        
        # Add intercept term
        X_train = np.ones((n_glaciers, n_features+1))
        X_train[:, 1:] = X_all[mask]
        y_train = y_true[mask]
        
        # Regularized matrix solve
        XtX = X_train.T @ X_train
        XtX += np.eye(XtX.shape[0]) * 1e-8
        beta = np.linalg.solve(XtX, X_train.T @ y_train)
        
        # Prediction
        X_test = np.ones(n_features+1)
        X_test[1:] = X_all[i]
        residuals[i] = y_true[i] - (X_test @ beta)
    
    return np.sqrt(np.mean(residuals**2))


from numba import njit
import numpy as np

@njit
def Validate_U_numba_batched(P_indices,X_p, y_p,u_obs):
    """
    Optimized Numba version
    Args:
    - X_all: 2D array (n_glaciers, n_features) of predictors
    - y_true: 1D array of true mean wind speeds
    - n_glaciers: Total number of glaciers
    """
    n_glaciers = X_p.shape[0]
    total_error = 0.0
    valid_glaciers = 0
    
    for i in range(n_glaciers):
        mask = np.ones(n_glaciers, dtype=np.bool_)
        mask[i] = True
        X_train = np.hstack((np.ones((n_glaciers, 1)), X_p[mask][:, P_indices]))
        y_train = y_p[mask]
 
        # Regularized matrix solve
        XtX_D = X_train.T @ X_train
        Xty_D = X_train.T @ y_train
        beta = np.linalg.solve(XtX_D, Xty_D)
        # Prediction
        u_pred = beta[0] + np.sum(X_p[i, P_indices] * beta[1:])
        total_error += (u_obs[i] - u_pred)**2
        valid_glaciers += 1

    
    return np.sqrt(total_error / (valid_glaciers))
