import os, shutil, time

datasets = ['hprd', 'patents', 'yeast', 'youtube']
query_sizes = [4, 8, 16, 24, 32]
models = ['GCN', 'GIN', 'GAT', 'GraphSAGE']
# datasets = ['hprd']
# query_sizes = [4]
# models = ['GCN']

# rm_saved = True
rm_saved = False
saved_folder = 'saved'
if rm_saved and os.path.exists(saved_folder):
    shutil.rmtree(saved_folder)


def exp(time_file, datasets, query_sizes, models, layers):
    res=''
    for d in datasets:
        for q in query_sizes:
            for arch in models:
                for layer in layers:
                    # train
                    cmd = 'python3 train.py -c config.json --a {} --d {} --q {} --k {}'.format(arch, d, q, layer)
                    res+='{}'.format(cmd)+'\n'
                    tb=time.perf_counter()
                    os.system(cmd)
                    te=time.perf_counter()
                    res+='train_time (ms): {:.2f}'.format((te-tb)*1000)+'\n'

                    # test
                    model_checkpoint_folder = 'saved/models/{}_k{}_{}_Q{}'.format(arch, layer, d, q)
                    for timestamp in os.listdir(model_checkpoint_folder):
                        if 'model_best.pth' in os.listdir(model_checkpoint_folder+'/'+timestamp):
                            ptn = model_checkpoint_folder+'/'+timestamp+'/model_best.pth'
                            cmd = 'python3 test.py --resume {} --a {} --d {} --q {} --k {}'.format(ptn, arch, d, q, layer)
                            # DEBUG
                            print(cmd)
                            res+='{}'.format(cmd)+'\n'
                            tb=time.perf_counter()
                            os.system(cmd)
                            te=time.perf_counter()
                            res+='test_time (ms): {:.2f}'.format((te-tb)*1000)+'\n'
    with open(time_file, 'w') as fout:
        fout.write(res)

def exp1(time_file):
    exp(time_file=time_file,
        datasets=datasets,
        query_sizes=query_sizes,
        models=models,
        layers=[3])
    
def exp2(time_file):
    d_hardest = 'patents'
    q_hardest = 32
    exp(time_file=time_file,
        datasets=[d_hardest],
        query_sizes=[q_hardest],
        models=models,
        layers=[1, 2])

def test_log_2_rerun_test(log_file='q_error.txt'):
    run_instead_of_just_print_cmd = False
    def parse_setting(line):
        # Loading checkpoint: saved/models/GraphSAGE_k3_hprd_Q4/0523_113747/model_best.pth ...
        s = line.strip().split()[2].split('/')[2] # GraphSAGE_k3_hprd_Q4
        # a k d q
        settings = s.split('_')
        settings[1] = int(settings[1].strip('k'))
        settings[3] = int(settings[3].strip('Q'))
        return settings
    
    key = 'Loading checkpoint'
    res = ''
    with open(log_file) as fin:
        line = fin.readline()
        while(line):
            if key in line:
                [arch, layer, d, q] = parse_setting(line)
                line2 = fin.readline()
                try:
                    # {'loss': 0.6484649697209534, 'log_q_error_range': 2.4983410835266113, 'log_q_error_median': 0.022949039936065674, 'log_q_error_95': 1.0198538303375244, 'log_q_error_75': 0.6637955904006958, 'log_q_error_25': -0.2918395400047302, 'log_q_error_5': -1.4784871339797974, 'log_q_error_max': 1.161881685256958, 'log_q_error_min': -1.8584966659545898}
                    eval(line2)
                except:
                    model_checkpoint_folder = 'saved/models/{}_k{}_{}_Q{}'.format(arch, layer, d, q)
                    for timestamp in os.listdir(model_checkpoint_folder):
                        if 'model_best.pth' in os.listdir(model_checkpoint_folder+'/'+timestamp):
                            ptn = model_checkpoint_folder+'/'+timestamp+'/model_best.pth'
                            cmd = 'python3 test.py --resume {} --a {} --d {} --q {} --k {}'.format(ptn, arch, d, q, layer)
                            res+='{}'.format(cmd)+'\n'
                            if run_instead_of_just_print_cmd:
                                tb=time.perf_counter()
                                os.system(cmd)
                                te=time.perf_counter()
                                res+='test_time (ms): {:.2f}'.format((te-tb)*1000)+'\n'
            line = fin.readline()
    print(res)

if __name__ == '__main__':
    log_folder = 'exp_log'
    os.makedirs(log_folder, exist_ok=True)
    exp1(log_folder+'/exp1_time.txt')
    exp2(log_folder+'/exp2_time.txt')

    # test_log_2_rerun_test()
