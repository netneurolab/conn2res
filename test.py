import os
import numpy as np
import pandas as pd
from sklearn.base import is_classifier
from conn2res.tasks import Conn2ResTask
from conn2res.connectivity import Conn
from conn2res.reservoir import EchoStateNetwork
from conn2res.reservoir import MSSNetwork
from conn2res.readout import Readout
from conn2res import plotting, readout
import pickle
import time
from joblib import Parallel, delayed

TASKS = ['MemoryCapacity',]

CLASS_METRICS = ['mean_squared_error','corrcoef',]

ALPHAS = np.linspace(start=0,stop=2,num=11,endpoint=True)

PROJ_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJ_DIR, 'figs')

if not os.path.isdir(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

N_RUNS = 1


with open('/home/sasham/Desktop/data/SBMS_Matrices/sbms.pickle','rb') as file:
    matrix_dict = pickle.load(file) #dictionary 'dict' object

matrices_1st = matrix_dict.get('1') # 'list' objects
matrices_2n = matrix_dict.get('2')
matrices_3rd = matrix_dict.get('3')

mapping = np.load('/home/sasham/Desktop/data/SBMS_Matrices/module_mappings.npy', mmap_mode='r+')
#Possibly pass file object to boost speed


#This loops through all possible pairs of regions/modules
for i in range(np.amax(mapping)-1):
    for j in range(i+1, np.amax(mapping)):
        in_nodes_from = np.where(mapping == i)
        out_nodes_from = np.where(mapping == j)
         

for task_name in TASKS:
    print(f'\n------------------TASK: {task_name.upper()}--------------')

    OUTPUT_DIR = os.path.join(PROJ_DIR,'figs',task_name)
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    task = Conn2ResTask(name=task_name,)

    conn = Conn(w=matrices_1st[0])
    conn.scale_and_normalize()

    

    df_runs = []
    for run in range(N_RUNS):
        print('\n\t------------ run: {run} ------')

        x,y = task.fetch_data(n_trials=500, input_gain=1,horizon_max=-5)
        print("INPUT (x):\n",x,"\n",x.shape)
        print("\nOUTPUT (y):\n",y,"\n\n")
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
        for i in range(np.amax(mapping)):
            gr_nodes.append(conn.get_nodes('random',np.where(mapping == i),gr_nodes,n_nodes=1)[0])

        ext_nodes = conn.get_nodes('random', nodes_from=np.where(mapping == np.amin(mapping)),nodes_without=gr_nodes,n_nodes=task.n_features)

        int_nodes = conn.get_nodes('all',nodes_without=np.union1d(gr_nodes,ext_nodes))

        output_nodes = conn.get_nodes('random',
                                      nodes_from = np.intersect1d(int_nodes,np.where(mapping == np.amax(mapping))),
                                      n_nodes=task.n_features
                                    )

        mssn = MSSNetwork(
            w=conn.w,
            int_nodes=int_nodes,
            ext_nodes =ext_nodes,
            gr_nodes=gr_nodes,
            mode='forward'
        )

        readout_module = Readout(estimator=readout.select_model(y))

        metrics = CLASS_METRICS

        df_alpha = []
        for a in ALPHAS:
            print(f'\n\n\t------ alpha = {a} ------')

            mssn.w = a*conn.w

            start1 = time.time()
            rs_train = mssn.simulate(Vext=x_train)
            point = time.time()
            rs_test = mssn.simulate(Vext=x_test)
            end1 = time.time()

            print("\n\nSimulate time for rs_train: ",point - start1)
            print("Simulate time for rs_test: ",end1 - point)
            print("Simulate time elapsed (seconds): ",end1-start1,"\n\n")

            if run == 0 and a == 1.0:
                plotting.plot_reservoir_states(
                    x=x_train, reservoir_states=rs_train,
                    title=task.name,
                    savefig=True,
                    fname=os.path.join(OUTPUT_DIR, f'res_states_train_{task.name}'),
                    rc_params={'figure.dpi':300,'savefig.dpi':300},
                    show = True
                )
                plotting.plot_reservoir_states(
                    x=x_test, reservoir_states=rs_test,
                    title=task.name,
                    savefig=True,
                    fname=os.path.join(OUTPUT_DIR, f'res_states_test_{task.name}'),
                    rc_params={'figure.dpi':300,'savefig.dpi':300},
                    show = True
                )

            start2 = time.time()
            df_res = readout_module.run_task(
                X=(rs_train,rs_test),y=(y_train,y_test),
                sample_weight='both', metric=metrics,
                readout_modules=None, readout_nodes=None,
            )
            end2= time.time()

            print("Run_task time elapsed (seconds): ",end2-start2)

            df_res['alpha'] = np.round(a,3)
            df_alpha.append(df_res)

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
            fname=os.path.join(OUTPUT_DIR, f'perf_{task.name}_{metric}'),
            rc_params={'figure.dpi':300,'savefig.dpi':300},
            show=True
        )