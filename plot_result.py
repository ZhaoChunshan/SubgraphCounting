import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import numpy as np

models = ['GCN', 'GIN', 'GAT', 'GraphSAGE']
datasets = ['hprd', 'patents', 'yeast', 'youtube']
query_sizes = [4, 8, 16, 24, 32]
layers = [i for i in range(1, 4)]

beautiful_color = [
    'pink',
    'paleturquoise',
    'skyblue',
    'lightcoral', 
    'gold', 
    'coral', 
    'palegreen', 
    'plum', 
    'hotpink', 
]

plot_time_unit = 's'
def log_time_2_plot_time(t): # t is in ms
    if plot_time_unit == 's':
        return t/1000
    return t 

def hardest_case(f, interest_ds=[]):
    def parse_setting(line):
        # python3 test.py --resume saved/models/GCN_k3_hprd_Q4/0523_113722/model_best.pth --a GCN --d hprd --q 4
        lst = line.strip().split()
        # a d q
        return lst[-5], lst[-3], lst[-1]
    def parse_time(line):
        # train_time (ms): 22417.30
        return float(line.strip().split()[-1])
    labels = ['train', 'test']
    interest_ds_time_settings = [[{},{}]]*len(interest_ds)
    for i, _ in enumerate(interest_ds):
        for model in models:
            interest_ds_time_settings[i][0][model] = []
            interest_ds_time_settings[i][1][model] = []
    max_time_setting = [{}, {}]
    for model in models:
        max_time_setting[0][model] = [0, '', '']
        max_time_setting[1][model] = [0, '', '']
    with open(f) as fin:
        line = fin.readline()
        while(line):
            for i in range(2):
                a,d,q = parse_setting(line)
                line = fin.readline()
                t = parse_time(line)
                if t > max_time_setting[i][a][0]:
                    max_time_setting[i][a] = [t, d, q]
                if d in interest_ds:
                    pos = interest_ds.index(d)
                    interest_ds_time_settings[pos][i][a].append([t, q])
                line = fin.readline()

    for j, d in enumerate(interest_ds):
        print(d)
        for i in range(2):
            for model in models:
                print('\t{} {}'.format(labels[i], model))
                for [t, q] in interest_ds_time_settings[j][i][model]:
                    print('\tq: {} time (ms): {}'.format(q, t))
    print()
    for i in range(2):
        print(labels[i])
        for model in models:
            [t, d, q] = max_time_setting[i][model]
            print('model: {} max_time (ms): {} d: {} q: {}'.format(model, t, d, q))

# dqka2qerror: {}
def test_log_2_q_error(log_file, dqka2qerror = {}):
    def parse_setting(line):
        # Loading checkpoint: saved/models/GraphSAGE_k3_hprd_Q4/0523_113747/model_best.pth ...
        s = line.strip().split()[2].split('/')[2] # GraphSAGE_k3_hprd_Q4
        # a k d q
        settings = s.split('_')
        settings[1] = int(settings[1].strip('k'))
        settings[3] = int(settings[3].strip('Q'))
        return settings
    def parse_q_error(line):
        # {'loss': 0.6484649697209534, 'log_q_error_range': 2.4983410835266113, 'log_q_error_median': 0.022949039936065674, 'log_q_error_95': 1.0198538303375244, 'log_q_error_75': 0.6637955904006958, 'log_q_error_25': -0.2918395400047302, 'log_q_error_5': -1.4784871339797974, 'log_q_error_max': 1.161881685256958, 'log_q_error_min': -1.8584966659545898}
        dct = eval(line)
        return {
            'whislo': dct['log_q_error_5'],    # whisker low
            'q1': dct['log_q_error_25'], # first quartile
            'med': dct['log_q_error_median'],        # median
            'q3': dct['log_q_error_75'], # third quartile
            'whishi': dct['log_q_error_95'],    # whisker high
            'fliers': []          # outliers
        }
    
    key = 'Loading checkpoint'
    with open(log_file) as fin:
        line = fin.readline()
        while(line):
            if key in line:
                try:
                    [a, k, d, q] = parse_setting(line)
                    line2 = fin.readline()
                    q_error = parse_q_error(line2)
                except:
                    print(line, line2)
                else:
                    if d not in dqka2qerror:
                        dqka2qerror[d]= {}
                    if q not in dqka2qerror[d]:
                        dqka2qerror[d][q] = {}
                    if k not in dqka2qerror[d][q]:
                        dqka2qerror[d][q][k] = {}
                    if a in dqka2qerror[d][q][k]:
                        print('ERROR! {} Q{} k{} {} already met!'.format(d, q, k, a))
                    dqka2qerror[d][q][k][a] = q_error
            line = fin.readline()
    return dqka2qerror

# 不同的图：不同的数据集
# 横轴：查询大小
# 多个bar：不同网络
# 纵轴：q_error（箱线图）
def exp1_plot_q_error(dqka2qerror, plot_output_folder):
    os.makedirs(plot_output_folder, exist_ok=True)
    k = 3
    dpi = 600
    xlb = 'query size'
    ylb = 'error'
    for d in datasets:
        # a plot
        title = d
        plot_file = plot_output_folder + '/{}.jpg'.format(d)
        fig, ax = plt.subplots()

        xtick_positions = []
        xtick_labels = []
        pos = 1
        boxs = [] # box plot of each group
        for q in query_sizes:
            # a bar group
            data = []
            positions = []
            pos_b = pos
            for a in models: # a bar
                data.append(dqka2qerror[d][q][k][a])
                positions.append(pos)
                pos+=1
            box = ax.bxp(data, positions=positions, patch_artist=True)
            boxs.append(box)
            # set color
            for patch, color in zip(box['boxes'], beautiful_color):
                patch.set_facecolor(color)
            
            xtick_positions.append((pos_b+pos)//2) # mid of the group
            xtick_labels.append(str(q))
            pos+=1 # separate each group
        
        # set x labels
        ax.set_xticks(xtick_positions) 
        ax.set_xticklabels(xtick_labels)

        # set legend
        legend_patchs = []
        for i, a in enumerate(models):
            legend_patchs.append(mpatches.Patch(color=beautiful_color[i], label=a))    
        # plt.legend(handles=legend_patchs, loc='upper right')
        plt.legend(handles=legend_patchs)

        ax.set_title(title)
        ax.set_xlabel(xlb)
        ax.set_ylabel(ylb)  

        plt.savefig(plot_file, dpi=dpi)

# dqka2time = [{}, {}] # train test
def time_log_2_time(f, dqka2time = [{}, {}]):
    def parse_setting(line):
        default_k = 3
        # ... --a GCN --d hprd --q 4 
        # ... --a GCN --d hprd --q 4 --k 3
        lst = line.strip().split()
        # a k d q
        if lst[-2]=='--k':
            return [lst[-7], int(lst[-1]), lst[-5], int(lst[-3])]
        return [lst[-5], default_k, lst[-3], int(lst[-1])]
    def parse_time(line):
        # train_time (ms): 22417.30
        return float(line.strip().split()[-1])
    
    with open(f) as fin:
        line = fin.readline()
        while(line):
            for i in range(2):
                [a, k, d, q] = parse_setting(line)
                line = fin.readline()
                t = parse_time(line)
                if d not in dqka2time[i]:
                    dqka2time[i][d]= {}
                if q not in dqka2time[i][d]:
                    dqka2time[i][d][q] = {}
                if k not in dqka2time[i][d][q]:
                    dqka2time[i][d][q][k] = {}
                if a in dqka2time[i][d][q][k]:
                    print('ERROR! {} Q{} k{} {} already met! old: {} new: {}'.format(d, q, k, a, dqka2time[i][d][q][k][a], t))
                dqka2time[i][d][q][k][a] = t
                line = fin.readline()
    return dqka2time

# exp1_plot_q_error把纵轴改时间，箱线图改柱状图
def exp1_plot_time(dqka2time, plot_output_folder):
    os.makedirs(plot_output_folder, exist_ok=True)
    k = 3
    dpi = 600
    xlb = 'query size'
    ylb = 'time ({})'.format(plot_time_unit)
    time_kinds = ['train', 'inference']
    bar_width = 0.2
    for j, time_kind in enumerate(time_kinds):
        os.makedirs(plot_output_folder+'/'+time_kind, exist_ok=True)
        for d in datasets:
            # a plot
            title = d
            plot_file = plot_output_folder + '/{}/exp1_{}_{}.jpg'.format(time_kind, time_kind, d)
            fig, ax = plt.subplots()

            bar_num = len(query_sizes)
            x = np.arange(bar_num) # postion of each group
            x_begin = x - bar_width*bar_num/2
            max_height = -1
            min_height = -1
            # plot each bar
            for i, a in enumerate(models):
                data = []
                for q in query_sizes:
                    t = log_time_2_plot_time(dqka2time[j][d][q][k][a])
                    if (max_height > 0 and t > max_height) or max_height < 0:
                        max_height = t
                    if (min_height > 0 and t < min_height) or min_height < 0:
                        min_height = t
                    data.append(t)
                ax.bar(x=x_begin+i*bar_width, height=data, label=a, align='edge', width=bar_width, edgecolor=beautiful_color[i], color='white', hatch='\\\\\\')

            # set x labels
            ax.set_xticks(x) 
            ax.set_xticklabels(query_sizes)

            # set y lim
            ax.set_ylim([min_height/2, max_height+min_height/3])

            ax.set_title(title)
            ax.set_xlabel(xlb)
            ax.set_ylabel(ylb)  
            plt.legend()

            plt.savefig(plot_file, dpi=dpi)

# 最大的数据集，最难的查询
# 横轴：网络层数（k=1, 2, 3）
# 多个bar：不同网络
# 纵轴：q_error（箱线图）
def exp2_plot_q_error(dqka2qerror, plot_output_folder):
    os.makedirs(plot_output_folder, exist_ok=True)
    dpi = 600
    xlb = 'num of layer'
    ylb = 'error'
    d_hardest = 'patents'
    q_hardest = 32

    d = d_hardest
    q = q_hardest
    title = '{} Q{}'.format(d, q)
    plot_file = plot_output_folder + '/{}_Q{}.jpg'.format(d, q)
    # fig, ax = plt.subplots(figsize=(10, 6))
    fig, ax = plt.subplots()

    xtick_positions = []
    xtick_labels = []
    pos = 1
    boxs = [] # box plot of each group
    for k in layers:
        # a bar group
        data = []
        positions = []
        pos_b = pos
        for a in models: # a bar
            data.append(dqka2qerror[d][q][k][a])
            positions.append(pos)
            pos+=1
        box = ax.bxp(data, positions=positions, patch_artist=True)
        boxs.append(box)
        # set color
        for patch, color in zip(box['boxes'], beautiful_color):
            patch.set_facecolor(color)
        
        xtick_positions.append((pos_b+pos)//2) # mid of the group
        xtick_labels.append(str(k))
        pos+=1 # separate each group
    
    # set x labels
    ax.set_xticks(xtick_positions) 
    ax.set_xticklabels(xtick_labels)

    # set legend
    legend_patchs = []
    for i, a in enumerate(models):
        legend_patchs.append(mpatches.Patch(color=beautiful_color[i], label=a))    
    plt.legend(handles=legend_patchs, loc='lower right')
    # plt.legend(handles=legend_patchs)

    ax.set_title(title)
    ax.set_xlabel(xlb)
    ax.set_ylabel(ylb)  

    plt.savefig(plot_file, dpi=dpi)

# exp2_plot_q_error把纵轴改时间，箱线图改柱状图
def exp2_plot_time(dqka2time, plot_output_folder):
    os.makedirs(plot_output_folder, exist_ok=True)
    k = 3
    dpi = 600
    xlb = 'num of layer'
    ylb = 'time ({})'.format(plot_time_unit)
    d_hardest = 'patents'
    q_hardest = 32

    d = d_hardest
    q = q_hardest
    time_kinds = ['train', 'inference']
    bar_width = 0.2
    for j, time_kind in enumerate(time_kinds):
        title = '{} {} Q{}'.format(time_kind, d, q)
        os.makedirs(plot_output_folder+'/'+time_kind, exist_ok=True)
        plot_file = plot_output_folder + '/{}/exp2_{}_{}_Q{}.jpg'.format(time_kind, time_kind, d, q)
        fig, ax = plt.subplots()

        bar_num = len(layers)
        x = np.arange(bar_num) # postion of each group
        x_begin = x - bar_width*bar_num/2
        max_height = -1
        min_height = -1
        # plot each bar
        for i, a in enumerate(models):
            data = []
            for k in layers:
                t = log_time_2_plot_time(dqka2time[j][d][q][k][a])
                if (max_height > 0 and t > max_height) or max_height < 0:
                    max_height = t
                if (min_height > 0 and t < min_height) or min_height < 0:
                    min_height = t
                data.append(t)
            ax.bar(x=x_begin+i*bar_width, height=data, label=a, align='edge', width=bar_width, edgecolor=beautiful_color[i], color='white', hatch='\\\\\\')

        # set x labels
        ax.set_xticks(x) 
        ax.set_xticklabels(layers)

        # set y lim
        ax.set_ylim([min_height/2, max_height+min_height/3])

        ax.set_title(title)
        ax.set_xlabel(xlb)
        ax.set_ylabel(ylb)  
        plt.legend()

        plt.savefig(plot_file, dpi=dpi)

def plot_result():
    plot_output_folder = 'result_plot/'
    log_folder = 'exp_log'
    os.makedirs(log_file, exist_ok=True)

    # python3 test_all_model.py > exp_log/q_error.txt
    # 输出exp1_time.txt和exp2_time.txt
    exp1_time_file=log_folder+'/exp1_time.txt'
    log_file=log_folder+'/q_error.txt' 
    exp2_time_file=log_folder+'/exp2_time.txt'

    dqka2qerror = test_log_2_q_error(log_file)
    exp1_plot_q_error(dqka2qerror, plot_output_folder+'exp1')
    exp2_plot_q_error(dqka2qerror, plot_output_folder+'exp2')

    dqka2time = time_log_2_time(exp1_time_file)
    exp1_plot_time(dqka2time, plot_output_folder+'time')
    dqka2time = time_log_2_time(exp2_time_file, dqka2time)
    exp2_plot_time(dqka2time, plot_output_folder+'time')

if __name__ == '__main__':
    # hardest_case(exp1_time_file, ['patents'])
    plot_result()

