import os
import os.path as osp
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.style as style
style.use('ggplot')
plt.rcParams['axes.facecolor']='w'
plt.rcParams['axes.edgecolor']='black'
plt.rcParams['grid.color']='#abbbc6'
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, AutoMinorLocator
import seaborn as sns
import numpy as np
import string

axis_font_size = 16
title_font_size = 18

df = pd.read_csv('datasets_to_use.csv')
dna_datasets = sorted(list(df[df['type'] == 'dna']['Dataset'].values))
prot_datasets = sorted(list(df[df['type'] == 'protein']['Dataset'].values))
nlp_datasets = ['AIMed', 'BioInfer', 'CC1-LLL', 
    'CC2-IEPA', 'CC3-HPRD50', 'DrugBank', 'MedLine']
datasets = prot_datasets + dna_datasets + nlp_datasets

def vary_num_threads(csv, dataset):
    '''Train kernel computation time as the number
    of threads varies.
    '''
    outfile = dataset + "_threads.pdf"
    df = pd.read_csv(csv)
    threads = list(range(1, 21))

    exact = list(df['fastsk_exact_time'])
    approx = list(df['fastsk_approx_time'])
    approx_t1 = list(df['fastsk_approx_time_t1'])
    i50 = list(df['fastsk_I50'])
    gkm = list(df['gkm_time'])
    
    fig, ax = plt.subplots()
    ax.plot(threads, exact, label='FastGSK-Exact (20 threads)', marker='x', color='b')
    ax.plot(threads, i50, label='FastGSK-Approx (1 thread)', marker='*', color='g')
    #ax.plot(threads, approx_t1, label='FastSK (approx, 1 thread)', marker='x', color='orange')
    ax.plot(threads, gkm, label='gkmSVM-2.0 (20 threads)', color='r', marker='o')

    ax.set_xlabel("Number of Threads", size=axis_font_size)
    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_ylabel("Time (s)", size=axis_font_size)
    ax.set_ylim(bottom=0)
    #ax.set_yscale('log')
    ax.set_title("Kernel Computation Time vs Number of Threads ({})".format(dataset))
    ax.legend(prop={'size': 15})

    fig.tight_layout()
    plt.savefig(outfile)

def vary_g(csv, dataset):
    '''Train kernel computation time as the overall feature
    length varies
    '''
    outfile = dataset + "_g.pdf"
    df = pd.read_csv(csv)

    g = list(df['g'])
    m = list(df['m'])
    exact = list(df['fastsk_exact_time'])
    approx = list(df['fastsk_approx_time'])
    approx_t1 = list(df['fastsk_approx_time_t1'])
    gkm = list(df['gkm_time'])
    
    fig, ax = plt.subplots()
    ax.plot(g, exact, label='FastSK (exact)', marker='x', color='b')
    ax.plot(g, approx, label='FastSK (approx)', marker='x', color='g')
    ax.plot(g, approx_t1, label='FastSK (approx, 1 thread)', marker='x', color='orange')
    ax.plot(g, gkm, label='gkmSVM-2.0', color='r', marker='x')

    ax.set_xlabel(r'Feature length $g$')
    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_yscale('log')
    ax.set_ylabel("Time (s)")
    #ax.set_ylim(bottom=0)
    ax.set_title(r'Kernel Computation Time vs $g$ ({})'.format(dataset))
    ax.legend(prop={'size': 15})

    fig.tight_layout()
    plt.savefig(outfile)

def vary_I(csv, dataset):
    '''Train kernel computation time as the overall feature
    length varies
    '''
    outfile = dataset + "_I.pdf"
    df = pd.read_csv(csv)

    I = list(df['I'])
    acc = [100 * val for val in list(df['acc'])]
    auc = [100 * val for val in list(df['auc'])]
    
    fig, ax = plt.subplots()
    ax.plot(I, acc, label='Accuracy', marker='x', color='b')
    ax.plot(I, auc, label='AUC', marker='x', color='r')

    ax.set_xlabel(r'Maximum Number of Iterations')
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.xaxis.set_minor_locator(MultipleLocator(5))
    ax.tick_params(axis='both', which='major', labelsize=14)
    #ax.set_yscale('log')
    ax.set_ylabel("Performance")
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)
    ax.set_title('Varying Max Iters ({})'.format(dataset))
    ax.legend(prop={'size': 15})

    fig.tight_layout()
    plt.savefig(outfile)

def vary_delta(csv, dataset):
    '''Train kernel computation time as the overall feature
    length varies
    '''
    outfile = dataset + "_delta.pdf"
    df = pd.read_csv(csv)

    delta = list(df['delta'])
    print(delta)
    acc = [100 * val for val in list(df['acc'])]
    auc = [100 * val for val in list(df['auc'])]
    
    fig, ax = plt.subplots()
    ax.plot(delta, acc, label='Accuracy', marker='x', color='b')
    ax.plot(delta, auc, label='AUC', marker='x', color='r')

    ax.set_xlabel(r'$\delta$')
    #ax.xaxis.set_major_locator(MultipleLocator(0.005))
    ax.xaxis.set_minor_locator(MultipleLocator(0.01))
    ax.tick_params(axis='both', which='major', labelsize=14)
    #ax.set_yscale('log')
    ax.set_ylabel("Performance")
    ax.set_ylim(bottom=50)
    ax.set_xlim(left=0, right=1.1)
    ax.set_title(r'Varying $\delta$ ({})'.format(dataset))
    ax.legend(prop={'size': 15})

    fig.tight_layout()
    plt.savefig(outfile)

def big_fig_threads(csv_dir, outfile):
    files = os.listdir(csv_dir)
    num_files = len(files)
    rows, cols = 2, 5
    threads = list(range(1, 21))

    dna_files = sorted([f for f in files if f.split('_')[0] in dna_datasets])
    files = dna_files

    fig, axes = plt.subplots(rows, cols)
    fig.set_size_inches(25, 10)

    for r in range(rows):
        for c in range(cols):
            filenum = r * cols + c
            if (filenum >= num_files):
                break
            filename = os.fsdecode(files[filenum])
            dataset = filename.split('_')[0]
            if (dataset == 'EP300'):
                if filename.split('_')[1] == '47848':
                    dataset += '_47848'
            file = osp.join(csv_dir, filename)
            print(file)
            
            df = pd.read_csv(file)
            exact = list(df['fastsk_exact_time'])
            i50 = list(df['fastsk_I50'])
            approx = list(df['fastsk_approx_time_t1'])
            #approx_t1 = list(df['fastsk_approx_time_t1'])
            gkm = list(df['gkm_time'])
            
            ax = axes[r][c]
            
            ax.plot(threads, exact, label='FastGSK-Exact', color='b', marker='x')
            #ax.plot(threads, approx, label='FastSK (approx, 1 threa)', color='orange', marker='x')
            ax.plot(threads, i50, label='FastGSK-Approx (1 thread)', color='g', marker='x')
            ax.plot(threads, gkm, label='gkmSVM-2.0', color='r', marker='o')
            #ax.plot(threads, i50, label=r'FastSK ($\leq$ 50 iterations)', color='magenta', marker='x')

            ax.set_xlabel("Number of Threads", size=axis_font_size)
            ax.xaxis.set_major_locator(MultipleLocator(2))
            ax.xaxis.set_minor_locator(MultipleLocator(1))
            ax.tick_params(axis='both', which='major')
            if c == 0:
                ax.set_ylabel("Time (s)", size=axis_font_size)
            ax.set_ylim(bottom=0)
            #ax.set_yscale('log')
            ax.set_title("{}".format(dataset))
            ax.legend(prop={'size': 15})

            fig.tight_layout()
    
    plt.savefig(outfile)

def bigfig_I(csv_dir, outfile):
    files = os.listdir(csv_dir)
    num_files = len(files)
    rows, cols = 4, 5
    threads = list(range(1, 21))

    fig, axes = plt.subplots(rows, cols)
    fig.set_size_inches(25, 15)

    dna_files = sorted([f for f in files if f.split('_')[0] in dna_datasets])
    prot_files = sorted([f for f in files if f.split('_')[0] in prot_datasets])
    files = prot_files + dna_files

    for r in range(rows):
        for c in range(cols):
            filenum = r * cols + c

            if (filenum >= num_files):
                break
            filename = os.fsdecode(files[filenum])

            print(filename)
            dataset = filename.split('_')[0]
            if (dataset == 'EP300'):
                if filename.split('_')[1] == '47848':
                    dataset += '_47848'
            if (r <= 1):
                dataset += ' (protein)'
            else:
                dataset += ' (DNA)'
            file = osp.join(csv_dir, filename)
            
            df = pd.read_csv(file)

            I = list(df['I'])
            acc = [100 * val for val in list(df['acc'])]
            auc = [100 * val for val in list(df['auc'])]
            
            ax = axes[r][c]

            ax.plot(I, acc, label='Accuracy', marker='x', color='b')
            ax.plot(I, auc, label='AUC', marker='x', color='r')

            ax.set_xlabel(r'Positions Sampled (Iterations)', size=axis_font_size)
            #max_I = max(I)
            # if (max_I == 500):
            #     tick_spacing = 100
            # elif (200 <= max_I < 500):
            #     tick_spacing = 50
            # elif (100 <= max_I < 200):
            #     tick_spacing = 20
            # else:
            #     tick_spacing = 10
            #ax.xaxis.set_major_locator(MultipleLocator(tick_spacing))
            #ax.xaxis.set_minor_locator(MultipleLocator(tick_spacing // 2))
            ax.tick_params(axis='both', which='major', labelsize=axis_font_size)
            #ax.set_yscale('log')
            if c == 0:
                ax.set_ylabel("Performance", size=30)
            ax.set_ylim(top=101)
            #ax.set_xlim(left=0,)
            ax.set_title('{}'.format(dataset), size=title_font_size)
            ax.legend(prop={'size': 15})

            fig.tight_layout()
    
    plt.savefig(outfile)

def bigfig_g_time(csv_dir, outfile):
    files = os.listdir(csv_dir)
    rows, cols = 2, 5

    fig, axes = plt.subplots(rows, cols)
    fig.set_size_inches(25, 10)

    dna_files = sorted([f for f in files if f.split('_')[0] in dna_datasets])
    files = dna_files
    num_files = len(files)

    for r in range(rows):
        for c in range(cols):
            filenum = r * cols + c
            if (filenum >= num_files):
                break
            filename = os.fsdecode(files[filenum])
            print(filename)
            
            dataset = filename.split('_')[0]
            if (dataset == 'EP300'):
                if filename.split('_')[1] == '47848':
                    dataset += '_47848'
            file = osp.join(csv_dir, filename)
            
            df = pd.read_csv(file)
            g = list(df['g'])

            fastsk_approx_t1 = [v for v in list(df['FastSK-Approx 1 thread']) if v != 0]
            fastsk_approx_t10 = [v for v in list(df['FastSK-Approx 10 thread']) if v != 0]
            fastsk_approx_t20 = [v for v in list(df['FastSK-Approx 20 thread']) if v != 0]

            fastsk_approx_t1_no_var = [v for v in list(df['FastSK-Approx 1 thread no variance 50 iters']) if v != 0]
            fastsk_approx_t10_no_var = [v for v in list(df['FastSK-Approx 10 thread no variance 50 iters']) if v != 0]
            fastsk_approx_t20_no_var = [v for v in list(df['FastSK-Approx 20 thread no variance 50 iters']) if v != 0]

            gkm_approx_t1 = [v for v in list(df['gkm-Approx 1 thread']) if v != 0]
            gkm_approx_t10 = [v for v in list(df['gkm-Approx 10 thread']) if v != 0]
            gkm_approx_t20 = [v for v in list(df['gkm-Approx 20 thread']) if v != 0]

            ax = axes[r][c]
            
            ax.plot(g[:len(fastsk_approx_t1_no_var)], fastsk_approx_t1_no_var, label='FastGSK-Approx (1 thread)', color='b', marker='o')
            ax.plot(g[:len(fastsk_approx_t10_no_var)], fastsk_approx_t10_no_var, label='FastGSK-Approx (10 threads)', color='b', marker='x')
            ax.plot(g[:len(fastsk_approx_t20_no_var)], fastsk_approx_t20_no_var, label='FastGSK-Approx (20 threads)', color='b', marker='^')

            ax.plot(g[:len(gkm_approx_t1)], gkm_approx_t1, label='Gkm-Approx (1 thread)', color='r', marker='o')
            ax.plot(g[:len(gkm_approx_t10)], gkm_approx_t10, label='GkmSVM-2.0 (10 threads)', color='r', marker='x')
            ax.plot(g[:len(gkm_approx_t20)], gkm_approx_t20, label='GkmSVM-2.0 (20 threads)', color='r', marker='^')

            ax.set_xlabel(r'$g$', size=axis_font_size)
            ax.xaxis.set_major_locator(MultipleLocator(2))
            ax.xaxis.set_minor_locator(MultipleLocator(1))
            ax.tick_params(axis='both', which='major', labelsize=axis_font_size)
            if c == 0:
                ax.set_ylabel("Time (s)", size=30)
            #ax.set_ylim(bottom=0)
            ax.set_yscale('log')
            ax.set_title("{}".format(dataset), size=title_font_size)
            ax.legend(prop={'size': 11})

            fig.tight_layout()
    
    plt.savefig(outfile)

def get_speeds_over_g(csv_dir):
    files = os.listdir(csv_dir)

    dna_files = sorted([f for f in files if f.split('_')[0] in dna_datasets])
    for file in dna_files:
        file = osp.join(csv_dir, file)
        
        df = pd.read_csv(file)
        g = list(df['g'])
        exact = [v for v in list(df['fastsk_exact']) if v != 0]
        gkm_times = [v for v in list(df['gkm_exact']) if v != 0]
        approx_i50_times = list(df['fastsk_I50'])[:len(gkm_times)]

        speedups = [gkm / approx for (gkm, approx) in zip(gkm_times, approx_i50_times)]
        avg_speedup = sum(speedups) / len(speedups)
        max_speedup = max(speedups)
        #print('{} --> avg_speedup = {}'.format(file, avg_speedup))
        print('{}'.format(max_speedup))


def big_fig_delta(csv_dir, outfile):
    files = os.listdir(csv_dir)
    num_files = len(files)
    rows, cols = 4, 5
    threads = list(range(1, 21))

    fig, axes = plt.subplots(rows, cols)
    fig.set_size_inches(25, 15)

    dna_files = sorted([f for f in files if f.split('_')[0] in dna_datasets])
    prot_files = sorted([f for f in files if f.split('_')[0] in prot_datasets])
    files = prot_files + dna_files

    for r in range(rows):
        for c in range(cols):
            filenum = r * cols + c

            if (filenum >= num_files):
                break
            filename = os.fsdecode(files[filenum])

            print(filename)
            dataset = filename.split('_')[0]
            file = osp.join(csv_dir, filename)
            
            df = pd.read_csv(file)
            
            delta = list(df['delta'])
            acc = [100 * val for val in list(df['acc'])]
            auc = [100 * val for val in list(df['auc'])]
            
            ax = axes[r][c]
            
            ax.plot(delta, acc, label='Accuracy', marker='x', color='b')
            ax.plot(delta, auc, label='AUC', marker='x', color='r')

            ax.set_xlabel(r'$\delta$')
            #ax.xaxis.set_major_locator(MultipleLocator(0.005))
            ax.xaxis.set_minor_locator(MultipleLocator(0.01))
            ax.tick_params(axis='both', which='major', labelsize=14)
            #ax.set_yscale('log')
            ax.set_ylabel("Performance")
            ax.set_ylim(bottom=50)
            ax.set_xlim(left=0, right=1)
            ax.set_title(r'Varying $\delta$ ({})'.format(dataset))
            ax.legend(prop={'size': 15})
            fig.tight_layout()
    
    plt.savefig(outfile)

def increase_g(csv, dataset):
    outfile = dataset + '_increase_g.pdf'
    df = pd.read_csv(csv)
    
    g = list(df['g'])
    fastsk_auc = [100 * auc for auc in list(df['fastsk_approx_auc'])]
    gkm_auc = [100 * auc for auc in list(df['gkm_approx_auc'])]
    
    fig, ax = plt.subplots()
    
    ax.plot(g, fastsk_auc, label='AUC', marker='x', color='r')
    ax.plot(g, gkm_auc, label='Accuracy', marker='x', color='b')

    ax.set_xlabel(r'$g$', size=axis_font_size)
    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.tick_params(axis='both', which='major', labelsize=14)
    #ax.set_yscale('log')
    ax.set_ylabel("Performance", size=axis_font_size)
    ax.set_ylim(bottom=50, top=100)
    #ax.set_xlim(left=0, right=0.1)
    ax.set_title(r'Increasing $g$ ({})'.format(dataset))
    ax.legend(prop={'size': 15})
    fig.tight_layout()
    
    plt.savefig(outfile)

def increase_g_time(csv, dataset):
    outfile = dataset + "_g_time.pdf"
    
    df = pd.read_csv(csv)
    g = list(df['g'])
    exact = list(df['fastsk_exact'])
    gkm = [t for t in list(df['gkm_exact']) if t != 0]
    approx = list(df['fastsk_approx_t1'])
    
    fig, ax = plt.subplots()
    
    ax.plot(g, exact, label='FastGSK-Exact (20 threads)', color='b', marker='x')
    ax.plot(g, approx, label='FastGSK-Approx (1 thread)', color='g', marker='x')
    ax.plot(g[:len(gkm)], gkm, label='GkmSVM-2.0 (20 threads)', color='r', marker='o')

    ax.set_xlabel(r'Feature Length $g$', size=axis_font_size)
    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.tick_params(axis='both', which='major')
    ax.set_ylabel("Kernel Computation Time (s)", size=axis_font_size)
    #ax.set_ylim(bottom=0)
    ax.set_yscale('log')
    ax.set_title("Kernel Computation Time vs Feature Length ({})".format(dataset))
    ax.legend(prop={'size': 13})

    fig.tight_layout()
    print(outfile)
    plt.savefig(outfile)


def big_fig_increase_g(csv_dir, outfile):
    files = os.listdir(csv_dir)
    num_files = len(files)
    rows, cols = 2, 5
    threads = list(range(1, 21))

    fig, axes = plt.subplots(rows, cols)
    fig.set_size_inches(25, 7.5)

    dna_files = sorted([f for f in files if f.split('_')[0] in dna_datasets])
    prot_files = sorted([f for f in files if f.split('_')[0] in prot_datasets])
    files = prot_files + dna_files

    for r in range(rows):
        for c in range(cols):
            filenum = r * cols + c

            if (filenum >= num_files):
                break
            filename = os.fsdecode(files[filenum])

            print(filename)
            dataset = filename.split('_')[0]
            if (dataset == 'EP300'):
                if filename.split('_')[1] == '47848':
                    dataset += '_47848'
            file = osp.join(csv_dir, filename)
            
            df = pd.read_csv(file)
            
            g = list(df['g'])

            fastsk_approx_i50_auc = list(df['fastsk_approx_i50_auc'])
            fastsk_approx_conv_auc = list(df['fastsk_approx_conv_auc'])
            gkm_approx_auc = list(df['gkm_approx_auc'])
            
            ax = axes[r][c]

            ax.plot(g, fastsk_approx_conv_auc,
                label=r'FastSK-Approx ($\leq gCk$ iterations)',
                marker='x', 
                color='b')
            
            ax.plot(g, fastsk_approx_i50_auc,
                label='FastSK-Approx (50 iterations)', 
                marker='x', 
                color='orange')

            ax.plot(g, gkm_approx_auc,
                label=r'gkmSVM-2.0-Approx ($m_{max} = 3$)',
                marker='o',
                color='r')

            ax.set_xlabel(r'$g$', size=30)
            ax.xaxis.set_major_locator(MultipleLocator(2))
            ax.xaxis.set_minor_locator(MultipleLocator(1))
            ax.tick_params(axis='both', which='major', labelsize=axis_font_size)
            if (c == 0):
                ax.set_ylabel("AUC", size=30)
                ax.legend(prop={'size': 15})
            ax.set_ylim(bottom=0, top=1)
            ax.set_title(r'{}'.format(dataset), size=title_font_size)
            fig.tight_layout()
    
    print("Saving figures to {}".format(outfile))
    plt.savefig(outfile)

def nlp_perfs(csv):
    df = pd.read_csv(csv)
    fastsk_acc = list(df['fastsk_acc'])
    fastsk_auc = list(df['fastsk_auc'])
    lstm_acc = list(df['lstm_acc'])
    lstm_auc = list(df['lstm_auc'])
    blended_acc = list(df['blended_acc'])
    blended_auc = list(df['blended_auc'])

    avg_improvement_blended = [fast - blend for (fast, blend) in zip(fastsk_auc, blended_auc)]
    avg_improvement_blended = sum(avg_improvement_blended) / len(fastsk_auc)
    print("Avg AUC improvement over Blended = ", avg_improvement_blended)

    avg_improvement_lstm = [fast - lstm for (fast, lstm) in zip(fastsk_auc, lstm_auc)]
    avg_improvement_lstm = sum(avg_improvement_lstm) / len(fastsk_auc)
    print("Avg AUC improvement over LSTM = ", avg_improvement_lstm)
    
    parity = np.linspace(0, 1)

    fig, axes = plt.subplots(1, 2, figsize=(11,4))

    axes[0].plot(parity, parity, color='r', zorder=1)
    axes[0].scatter(blended_auc, fastsk_auc, color='b', marker='x', zorder=2)
    axes[0].set_xlabel("Blended Spectrum Kernel AUC", size=axis_font_size)
    axes[0].xaxis.set_major_locator(MultipleLocator(0.2))
    axes[0].xaxis.set_minor_locator(MultipleLocator(0.1))
    axes[0].yaxis.set_major_locator(MultipleLocator(0.2))
    axes[0].yaxis.set_minor_locator(MultipleLocator(0.1))
    axes[0].set_ylabel("FastGSK AUC", size=axis_font_size)
    axes[0].set_title("FastGSK vs Blended Spectrum Kernel")
    axes[0].tick_params(axis='both', which='major', labelsize=14)
    axes[0].text(-0.1, 1.2, '(a)', transform=axes[0].transAxes,
      fontsize=title_font_size, va='top', ha='right')

    axes[1].plot(parity, parity, color='r', zorder=1)
    axes[1].scatter(lstm_auc, fastsk_auc, color='b', marker='x', zorder=2)
    axes[1].set_xlabel("LSTM AUC", size=axis_font_size)
    axes[1].set_title("FastGSK vs LSTM AUC (NLP data)")
    axes[1].tick_params(axis='both', which='major', labelsize=14)
    axes[1].xaxis.set_major_locator(MultipleLocator(0.2))
    axes[1].xaxis.set_minor_locator(MultipleLocator(0.1))
    axes[1].yaxis.set_major_locator(MultipleLocator(0.2))
    axes[1].yaxis.set_minor_locator(MultipleLocator(0.1))
    axes[1].text(-0.1, 1.2, '(b)', transform=axes[1].transAxes,
      fontsize=title_font_size, va='top', ha='right')
    
    fig.tight_layout()
    plt.savefig('nlp_auc_plot.pdf')

    # # vs lstm acc line plot
    # fig, ax = plt.subplots()
    # ax.scatter(lstm_acc, fastsk_acc, color='b', marker='x')
    # ax.plot(parity, parity, color='r', zorder=1)
    # ax.set_xlabel("LSTM Accuracy", size=axis_font_size)
    # ax.set_ylabel("FastGSK Accuracy", size=axis_font_size)
    # ax.set_title("FastGSK vs LSTM Accuracy (NLP Data)", size=title_font_size)
    # ax.tick_params(axis='both', which='major', labelsize=14)
    # fig.tight_layout()
    # plt.savefig('nlp_lstm_acc.pdf')

def lstm_proportions(csv, outfile):
    df = pd.read_csv(csv)
    size = np.array(df['num'])
    fastsk_auc = np.array(df['fastgsk_auc'])
    avg = np.array(df['avg'])
    stdev = np.array(df['stdev'])

    fig, ax = plt.subplots()
    
    ax.plot(size, fastsk_auc, label='FastGSK', marker='x', color='b')
    ax.plot(size, avg, label='LSTM', marker='o', color='r')
    ax.fill_between(size, avg - stdev, avg + stdev, alpha=0.5)

    ax.set_xlabel(r'Number of Training Samples', size=axis_font_size)
    ax.xaxis.set_major_locator(MultipleLocator(2000))
    ax.xaxis.set_minor_locator(MultipleLocator(1000))
    ax.tick_params(axis='both', which='major', labelsize=14)
    #ax.set_yscale('log')
    ax.set_ylabel("Test AUC", size=axis_font_size)
    ax.set_ylim(bottom=0.50, top=1)

    #ax.set_title(r'')
    ax.legend(prop={'size': 15})
    fig.tight_layout()
    plt.savefig(outfile)

def approx_vs_exact(csv, outprefix):
    '''Read fastsk summary data, plot comparisons of the
    exact vs approx algorithm
    '''

    df = pd.read_csv(csv)
    datasets = list(df['Dataset'])
    approx_auc = list(df['approx_auc'])
    approx_acc = list(df['approx_acc'])
    exact_auc = list(df['exact_auc'])
    exact_acc = list(df['exact_acc'])
    approx_time = list(df['approx_time_t1'])
    exact_time = list(df['exact_time'])
    exact_time_t1 = list(df['exact_time_t1'])
    i50_time = list(df['train_k_time_I50'])
    i50_acc = list(df['I50_acc'])
    i50_auc = list(df['I50_auc'])
    
    # convergence 
    avg_auc_loss = 100 * sum([a - e for a, e in zip(approx_auc, exact_auc)]) / len(exact_auc)
    avg_acc_loss = sum([a - e for a, e in zip(approx_acc, exact_acc)]) / len(exact_acc)
    print("Convergence: AUC loss: {}%, Acc loss: {}%".format(avg_auc_loss, avg_acc_loss))

    # 50 iters
    avg_auc_loss = 100 * sum([a - e for a, e in zip(i50_auc, exact_auc)]) / len(exact_auc)
    avg_acc_loss = sum([a - e for a, e in zip(i50_acc, exact_acc)]) / len(exact_acc)
    print("50 iters: AUC loss: {}%, Acc loss: {}%".format(avg_auc_loss, avg_acc_loss))

    # parity line
    parity = np.linspace(0, 1)

    # approx vs exact auc line plot
    fig, ax = plt.subplots()
    ax.scatter(exact_auc, i50_auc, color='b', marker='x',zorder=2)
    ax.plot(parity, parity, color='r', zorder=1)
    ax.set_xlabel("FastGSK-Exact AUC", size=axis_font_size)
    ax.set_ylabel("FastGSK-Approx AUC", size=axis_font_size)
    ax.set_title("FastGSK-Approx vs Exact (AUC)", size=title_font_size)
    ax.tick_params(axis='both', which='major', labelsize=14)
    fig.tight_layout()
    plt.savefig(outprefix + '_approx_vs_exact_auc.pdf')

    # approx vs exact train kernel time line plot
    time_parity = np.linspace(0, 20000)
    fig, ax = plt.subplots()
    #exact_time = [20 * t for t in exact_time]
    ax.scatter(exact_time_t1, i50_time, color='b', marker='x')
    ax.plot(time_parity, time_parity, color='r', zorder=1)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel("FastGSK-Exact Time (s)", size=axis_font_size)
    ax.set_ylabel("FastGSK-Approx Time (s)", size=axis_font_size)
    ax.set_title("FastGSK-Approx vs Exact (Time)", size=title_font_size)
    ax.tick_params(axis='both', which='major', labelsize=14)
    fig.tight_layout()
    plt.savefig(outprefix + '_approx_vs_exact_time_line.pdf')

    # approx vs exact acc line plot
    # fig, ax = plt.subplots()
    # ax.scatter(exact_acc, i50_acc, color='b', marker='x')
    # ax.plot(parity, parity, color='r', zorder=1)
    # ax.set_xlabel("FastGSK-Exact Accuracy", size=axis_font_size)
    # ax.set_ylabel("FastGSK-Approx Accuracy", size=axis_font_size)
    # ax.set_title("FastGSK-Approx vs Exact (Accuracy)", size=title_font_size)
    # ax.tick_params(axis='both', which='major', labelsize=14)
    # fig.tight_layout()
    # plt.savefig(outprefix + '_approx_vs_exact_acc.pdf')

#approx_vs_exact('fastsk_summary_perfs_right_datasets.csv', 'i50')
#nlp_perfs('nlp_results.csv')
#big_fig_threads('./vary_thread_results', 'bigfig_threads_I50.pdf')
#vary_g('EP300_vary_g.csv', 'EP300')
#bigfig_g_time('./g_time', 'g_time.pdf')
#bigfig_g_time('./g_times_nov14', 'g_times_nov14.pdf')
#get_speeds_over_g('./g_time')
#vary_delta('1.1_vary_delta.csv', '1.1')
#bigfig_I('./vary_I', 'bigfig_I.pdf')
#big_fig_delta('./vary_delta_csvs', 'bigfig_delta.pdf')
#big_fig_increase_g('./increase_g', 'bigfig_increase_g.pdf')
#big_fig_increase_g('./increase_g_m4', 'bigfig_increase_g_m4.pdf')
#big_fig_increase_g('./increase_g_k6', 'bigfig_increase_g_k6.pdf')
#big_fig_increase_g('./k6C', 'k6C.pdf')
#big_fig_increase_g('./g_auc_k6', 'g_auc_k6.pdf')
#big_fig_increase_g('./increase_g_k4', 'protein_auc_k4.pdf')
big_fig_increase_g('./dec15_g_auc_results', 'dec15_g_auc_protein_results.pdf')
#vary_num_threads('./vary_thread_results/RAD21_vary_threads.csv', 'RAD21')
#increase_g('./increase_g_k6/RAD21_increase_g_k6.csv', 'RAD21')
#increase_g_time('./g_time_k6/CTCF_vary_g.csv', 'CTCF')
#increase_g_time('./g_time/CTCF_g_time.csv', 'CTCF')
#lstm_proportions('lstm_proportions2.csv', 'lstm_props.pdf')
