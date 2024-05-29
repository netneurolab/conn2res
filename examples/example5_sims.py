import os
import warnings
import numpy as np
import pandas as pd
from sklearn.base import is_classifier
from conn2res.conn2res.tasks import Conn2ResTask
from conn2res.conn2res.connectivity import Conn
from conn2res.conn2res.reservoir import MSSNetwork, MSSNetworkCupy
from conn2res.conn2res.readout import Readout
from conn2res.conn2res import plotting, readout
import pickle
import time
import cupy as cp
import multiprocessing as mp
import copy
import sys

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

#List of taks to do from conn2res.tasks
TASKS = ['MemoryCapacity',]

#metrics to measure
CLASS_METRICS = ['mean_squared_error','corrcoef',]

#Alpha values by which to scale connection matrix
ALPHAS = np.linspace(start=0,stop=2,num=11,endpoint=True)

#Directory for saving figures
PROJ_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJ_DIR, 'figs')

if not os.path.isdir(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

N_RUNS = 1

def run_experiment():
    #loads the pickle file from local
    #can be replaced with any type of file, but matrix extracted should be of type numpy.ndarray
    with open('/conntection_matrix','rb') as file:
        matrix = pickle.load(file) #dictionary 'dict' object

    #loads the node mappings for the current connection matrix
    mapping = np.load('/module_mappings.npy', mmap_mode='r+')

    #list of all possible regions
    mapping_unique = np.unique(mapping)

    for task_name in TASKS:
        print(f'\n------------------TASK: {task_name.upper()}--------------')

        OUTPUT_DIR = os.path.join(PROJ_DIR,'figs',task_name)
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)

        task = Conn2ResTask(name=task_name,)

        df_runs = []

        for run in range(N_RUNS):
            print(f'\n\t------------ run: {run+1} ------')
            conn = Conn(w=matrix)
            conn.scale_and_normalize()

            regions = []

            #establishes 1 ground node per regions defined in the mapping array
            gr_nodes = []
            for x in mapping_unique:
                gr_nodes.append(conn.get_nodes('random',np.where(mapping == x)[0],gr_nodes,n_nodes=1)[0])

            #arbitrary features values (placeholder, gets changed later)
            features = 1

            #arbitrary internal and external nodes set to enable initialization of base memristive network
            #internal and external node classifications are changed later to enable per region assignments
            ext_nodes = conn.get_nodes('random',nodes_without=gr_nodes,n_nodes=features)
            int_nodes = conn.get_nodes('all',nodes_without=np.union1d(gr_nodes,ext_nodes))

            #initializes base class of MSSNetwork so that initial properties of the network stay
            #consistent between different alpha runs and internal/external node classifications
            base = MSSNetworkCupy(
                w=conn.w,
                int_nodes=int_nodes,
                ext_nodes=ext_nodes,
                gr_nodes=gr_nodes,
                mode='forward'
            )


            #loops through all unique cortical regions defined in mapping to enable each region to contain
            #the input node(s)
            #Note: in MSSN, out is the reading across ALL internal nodes, but can be specified to include specific ones
            for i in mapping_unique:
                #fetches in/out data for memory capacity task
                #also returns to z, the input data that has been cut off

                #running simulate on z will account for the data needed to make the first
                #prediction in the output data according to each time lag 
                x,y,z = task.fetch_data(n_trials=500, input_gain=1,horizon_max=-5)

                if run == 0:
                        plotting.plot_iodata(
                            x=x, y=y, title=task_name, savefig=True, 
                            fname= os.path.join(OUTPUT_DIR, f'io_{task_name}'),
                            rc_params={'figure.dpi':300, 'savefig.dpi':300},
                            show = False
                        )

                x_train, x_test, y_train, y_test = readout.train_test_split(x,y)

                #create copy of base network
                mssn = copy.deepcopy(base)

                #defines internal and external nodes for this iteration of unique cortical region
                ext_nodes = conn.get_nodes('random', nodes_from=np.where(mapping == i)[0],nodes_without=gr_nodes,n_nodes=task.n_features)
                int_nodes = conn.get_nodes('all',nodes_without=np.union1d(gr_nodes,ext_nodes))

                #updates copied network with new node classifications
                mssn._I = cp.asarray(int_nodes)
                mssn._E = cp.asarray(ext_nodes)
                mssn._n_external_nodes = len(ext_nodes)
                mssn._n_internal_nodes = len(int_nodes)

                readout_module = Readout(estimator=readout.select_model(y))

                metrics = CLASS_METRICS

                df_alpha = []

                #lock to synchronize child processes for the appending to the dictionary
                lock = mp.Lock()

                #creates the dicrionary from mp.Manager to collect the output from each alpha run 
                manager = mp.Manager()
                return_dict = manager.dict()
                
                #creates a pool of processes, opted for this method instead of mp.Pool since synchronization was easier
                #Each process runs an individual iteration of an alpha value
                #(parallelizes across alpha value simulations)
                pool = [mp.Process(target=run_alpha,args=(a,lock,mssn,readout_module,z,x_train,x_test,y_train,y_test,df_alpha,return_dict)) for a in range(len(ALPHAS))]

                #runs all the Processes
                for p in pool:
                    p.start()
                    
                #Synchronizes, waits for all processes to end
                for p in pool:
                    p.join()
                    
                #adds results of each alpha run to df_alpha
                for a in range(len(ALPHAS)):
                    df_alpha.append(return_dict.get(a))

                df_alpha = pd.concat(df_alpha,ignore_index=True)
                df_alpha['run'] = run

                regions.append(df_alpha)
                    
            #find the average of each alpha value simulation across each cortical region as input        
            avrg = regions[0]

            for x in avrg.index.tolist():
                elements = np.array([df.at[x,'corrcoef'] for df in regions])
                avrg.at[x,'corrcoef'] = np.mean(elements)

            df_runs.append(avrg)

            ####################################
            df_runs = pd.concat(df_runs,ignore_index=True)
            if 'module' in df_runs.columns:
                df_subj = df_runs[
                    ['module','n_nodes','run', 'alpha'] + metrics
                ]
            else:
                df_subj = df_runs[
                    ['run', 'alpha'] + metrics
                ]          

            df_subj.to_csv(os.path.join(OUTPUT_DIR,f'results_{task.name}.csv'), index=False)

            for metric in metrics:
                plotting.plot_performance(
                    df_subj,x='alpha',y=metric,
                    title=task.name, savefig=True,
                    fname=os.path.join(OUTPUT_DIR, f'perf_{task.name}_{metric}'),
                    rc_params={'figure.dpi':300,'savefig.dpi':300},
                    show=False
                )


def run_alpha(a,l,mssn,w,readout_module,z_sliced,x_train, x_test, y_train, y_test,df_alpha,return_dict):
    time.sleep(1)
    mssnc = copy.deepcopy(mssn)
    print(f'\n\n\t------ alpha = {ALPHAS[a]} ------')
    metrics = CLASS_METRICS

    #updates copied network connection matrix with current alpha value
    mssnc._W = ALPHAS[a]*mssnc._W

    #simulates cut data to "initialize" the network before the first prediction
    mssnc.simulate(Vext=z_sliced)

    #simulates the network on the test and train data
    rs_train = mssnc.simulate(Vext=x_train)
    rs_test = mssnc.simulate(Vext=x_test)
    

    # if run == 0 and a == 1.0:
    #     plotting.plot_reservoir_states(
    #         x=x_train, reservoir_states=rs_train,
    #         title=task.name,
    #         savefig=True,
    #         fname=os.path.join(OUTPUT_DIR, f'res_states_train_{task.name}'),
    #         rc_params={'figure.dpi':300,'savefig.dpi':300},
    #         show = True
    #     )
    #     plotting.plot_reservoir_states(
    #         x=x_test, reservoir_states=rs_test,
    #         title=task.name,
    #         savefig=True,
    #         fname=os.path.join(OUTPUT_DIR, f'res_states_test_{task.name}'),
    #         rc_params={'figure.dpi':300,'savefig.dpi':300},
    #         show = True
    #     )


    df_res = readout_module.run_task(
        X=(rs_train,rs_test),y=(y_train,y_test),
        sample_weight='both', metric=metrics,
        readout_modules=None, readout_nodes=None,
    )

    df_res['alpha'] = np.round(ALPHAS[a],3)

    #locks critical region of code block to only allow 1 process to add to the dictionary at a time
    l.acquire()
    return_dict[a] = df_res
    l.release()

    return 0


def main():
    mp.set_start_method('spawn')
    run_experiment()


if __name__ == '__main__':
    main()            