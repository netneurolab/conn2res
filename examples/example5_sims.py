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

TASKS = ['MemoryCapacity',]

CLASS_METRICS = ['mean_squared_error','corrcoef',]

ALPHAS = np.linspace(start=0,stop=2,num=11,endpoint=True)

#Directory for saving figures
PROJ_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJ_DIR, 'figs')

if not os.path.isdir(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

N_RUNS = 20

def run_experiment():
    #loads the ickle dictionary of connectome matrices
    with open('/home/sasham/Desktop/data/SBMS_Matrices/sbms.pickle','rb') as file:
        matrix_dict = pickle.load(file) #dictionary 'dict' object

    #extracts the 3 matrix sets of levels of hierarchy 
    matrices_1st = matrix_dict.get('1') # 'list' objects
    matrices_2n = matrix_dict.get('2')
    matrices_3rd = matrix_dict.get('3')

    #loads the mapping (shared between these artificial connectomes)
    mapping = np.load('/home/sasham/Desktop/data/SBMS_Matrices/module_mappings.npy', mmap_mode='r+')

    mapping_unique = np.unique(mapping)

    for task_name in TASKS:
        print(f'\n------------------TASK: {task_name.upper()}--------------')

        OUTPUT_DIR = os.path.join(PROJ_DIR,'figs',task_name)
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)

        task = Conn2ResTask(name=task_name,)
        
        #loops through the 3 levels of hierarchy
        for lvl in range(3):
            print(f'\n\t------------ Level: {lvl+1} ---------')
            if(lvl==0): lvl_matrices = matrices_1st
            elif (lvl==1): lvl_matrices = matrices_2n
            else: lvl_matrices = matrices_3rd

            df_runs = []
            #runs: Currently the number of matrices to take from each level
            for run in range(N_RUNS):
                print(f'\n\t------------ run: {run+1} ------')
                conn = Conn(w=lvl_matrices[run])
                conn.scale_and_normalize()

                #loops through all the smallest level of modules and does a run where the ext node is in a specific module
                for i in mapping_unique:
                    x,y = task.fetch_data(n_trials=500, input_gain=1,horizon_max=-5)
                    if run == 0:
                            plotting.plot_iodata(
                                x=x, y=y, title=task_name, savefig=True, 
                                fname= os.path.join(OUTPUT_DIR, f'io_{task_name}'),
                                rc_params={'figure.dpi':300, 'savefig.dpi':300},
                                show = False
                            )

                    x_train, x_test, y_train, y_test = readout.train_test_split(x,y)

                    #Chooses 1 ground node per module/region
                    gr_nodes = []
                    for x in range(np.amax(mapping)):
                        gr_nodes.append(conn.get_nodes('random',np.where(mapping == x),gr_nodes,n_nodes=1)[0])

                    ext_nodes = conn.get_nodes('random', nodes_from=np.where(mapping == i),nodes_without=gr_nodes,n_nodes=task.n_features)

                    int_nodes = conn.get_nodes('all',nodes_without=np.union1d(gr_nodes,ext_nodes))

                    # output_nodes = conn.get_nodes('random',
                    #                             nodes_from = np.intersect1d(int_nodes,np.where(mapping == np.amax(mapping))),
                    #                             n_nodes=task.n_features
                    #                             )

                    mssn = MSSNetworkCupy(
                        w=conn.w,
                        int_nodes=int_nodes,
                        ext_nodes =ext_nodes,
                        gr_nodes=gr_nodes,
                        mode='forward'
                    )

                    readout_module = Readout(estimator=readout.select_model(y))

                    metrics = CLASS_METRICS

                    df_alpha = []

                    #lock to synchronize child processes for the appending to the dictionary
                    lock = mp.Lock()

                    #creates the dicrionary from mp.Manager to collect the output from each alpha run 
                    manager = mp.Manager()
                    return_dict = manager.dict()
                    
                    #creates a pool of processes, opted for this method instead of mp.Pool since synchronization was easier
                    pool = [mp.Process(target=run_alpha,args=(a,lock,mssn,conn.w,readout_module,x_train,x_test,y_train,y_test,df_alpha,return_dict)) for a in range(len(ALPHAS))]

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

                    df_runs.append(df_alpha)
                    
    

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
                    fname=os.path.join(OUTPUT_DIR, f'perf_{task.name}_{metric}_{lvl+1}'),
                    rc_params={'figure.dpi':300,'savefig.dpi':300},
                    show=False
                )


def run_alpha(a,l,mssn,w,readout_module,x_train, x_test, y_train, y_test,df_alpha,return_dict):
    time.sleep(1)
    # print("\npid: ",os.getpid())
    mssnc = copy.deepcopy(mssn)
    print(f'\n\n\t------ alpha = {ALPHAS[a]} ------')
    metrics = CLASS_METRICS
    mssnc.w = ALPHAS[a]*w
    # with cp.cuda.Stream(non_blocking=True):
    start1 = time.time()
    rs_train = mssnc.simulate(Vext=x_train)
    point = time.time()
    rs_test = mssnc.simulate(Vext=x_test)
    end1 = time.time()

    print("\n\nSimulate time for rs_train: ",point - start1)
    print("Simulate time for rs_test: ",end1 - point)
    print("Simulate time elapsed (seconds): ",end1-start1,"\n\n")

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

    start2 = time.time()
    df_res = readout_module.run_task(
        X=(rs_train,rs_test),y=(y_train,y_test),
        sample_weight='both', metric=metrics,
        readout_modules=None, readout_nodes=None,
    )
    end2= time.time()

    print("Run_task time elapsed (seconds): ",end2-start2)

    df_res['alpha'] = np.round(ALPHAS[a],3)
    l.acquire()
    df_alpha.append(df_res)
    return_dict[a] = df_res
    l.release()

    return 0





def main():
    mp.set_start_method('spawn')
    run_experiment()


if __name__ == '__main__':
    main()            