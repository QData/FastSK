"""plot.py: Plotting code used for creating the figures in the FastSK paper.
See the accompanying README.md for instructions."""

__author__ = "Derrick Blakely"
__email__ = "dcb7xz@virginia.edu"
__date__ = "December 2019"

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
import matplotlib.ticker as mticker
import seaborn as sns
import numpy as np
import string
from scipy import special
import argparse

axis_font_size = 16
title_font_size = 18

df = pd.read_csv('datasets_to_use.csv')
dna_datasets = sorted(list(df[df['type'] == 'dna']['Dataset'].values))
prot_datasets = sorted(list(df[df['type'] == 'protein']['Dataset'].values))
nlp_datasets = ['AIMed', 'BioInfer', 'CC1-LLL', 
    'CC2-IEPA', 'CC3-HPRD50', 'DrugBank', 'MedLine']
datasets = prot_datasets + dna_datasets + nlp_datasets


def auc_summary_fig(csv='auc_results/acc_auc_summary_results.csv'):
    '''Create a figure summarizing the AUC results. Has 3 subplots:
    DNA, protein, and NLP. The plots each show FastSK-Approx vs
    several baseline models.
    '''
    df = pd.read_csv(csv)
    dna_df = df[df['type'] == 'dna']
    prot_df = df[df['type'] == 'protein']
    nlp_df = df[df['type'] == 'nlp']
    parity = np.linspace(0, 1)

    fig, axes = plt.subplots(1, 3)
    fig.set_size_inches(15, 5)

    ## DNA AUC plot
    fastsk_approx_dna = list(dna_df['fastsk-approx 50 iters auc'])
    gkm_exact_dna = list(dna_df['gkm-Exact auc'])
    lstm_dna = list(dna_df['lstm auc'])
    charcnn_dna = list(dna_df['charcnn auc'])
    gakco_dna = list(dna_df['gakco auc'])
    
    axes[0].scatter(gkm_exact_dna, fastsk_approx_dna,color='r',marker='o',label='gkmSVM-2.0')
    axes[0].scatter(gakco_dna, fastsk_approx_dna, color='g', marker='s', label='GaKCo')
    axes[0].scatter(lstm_dna, fastsk_approx_dna, color='b', marker='x', label='LSTM')
    axes[0].scatter(charcnn_dna, fastsk_approx_dna, color='orange', marker='*', label='Char-CNN')
    axes[0].plot(parity, parity, color='r', zorder=1)
    axes[0].set_xlabel("Baseline AUC", size=axis_font_size)
    axes[0].set_ylabel("FastSK AUC", size=axis_font_size)
    axes[0].set_title("DNA Test AUC", size=title_font_size)
    axes[0].text(-0.1, 1.1, '(a)', transform=axes[0].transAxes, fontsize=title_font_size, va='top', ha='right')
    axes[0].tick_params(axis='both', which='major', labelsize=14)
    axes[0].legend(prop={'size': 12}, loc='lower right', edgecolor='black')

    ## Protein AUC plot
    fastsk_approx_prot = list(prot_df['fastsk-approx 50 iters auc'])
    gkm_exact_prot = list(prot_df['gkm-Exact auc'])
    lstm_prot = list(prot_df['lstm auc'])
    charcnn_prot = list(prot_df['charcnn auc'])
    gakco_prot = list(dna_df['gakco auc'])
    
    axes[1].scatter(gkm_exact_prot, fastsk_approx_prot,color='r', marker='o', label='gkmSVM-2.0')
    axes[1].scatter(gakco_prot, fastsk_approx_dna, color='g', marker='s', label='GaKCo')
    axes[1].scatter(lstm_prot, fastsk_approx_prot, color='b', marker='x', label='LSTM')
    axes[1].scatter(charcnn_prot, fastsk_approx_prot, color='orange', marker='*', label='Char-CNN')
    axes[1].plot(parity, parity, color='r', zorder=1)
    axes[1].set_xlabel("Baseline AUC", size=axis_font_size)
    #axes[1].set_ylabel("FastSK AUC", size=axis_font_size)
    #axes[1].set_title("FastSK vs Baseline AUC (Protein)", size=title_font_size)
    axes[1].set_title("Protein Test AUC", size=title_font_size)
    axes[1].text(-0.1, 1.1, '(b)', transform=axes[1].transAxes, fontsize=title_font_size, va='top', ha='right')
    axes[1].tick_params(axis='both', which='major', labelsize=14)
    axes[1].legend(prop={'size': 12}, loc='lower right', edgecolor='black')

    ## Protein AUC plot
    fastsk_approx_nlp = list(nlp_df['fastsk-approx 50 iters auc'])
    blended_nlp = list(nlp_df['blended spectrum auc'])
    lstm_nlp = list(nlp_df['lstm auc'])
    charcnn_nlp = list(nlp_df['charcnn auc'])
    
    axes[2].scatter(blended_nlp, fastsk_approx_nlp,color='c', marker='^', label='Blended Spectrum Kernel')
    axes[2].scatter(lstm_nlp, fastsk_approx_nlp, color='r', marker='o', label='LSTM')
    axes[2].scatter(charcnn_nlp, fastsk_approx_nlp, color='orange', marker='*', label='Char-CNN')
    axes[2].plot(parity, parity, color='r', zorder=1)
    axes[2].set_xlabel("Baseline AUC", size=axis_font_size)
    axes[2].text(-0.1, 1.1, '(c)', transform=axes[2].transAxes, fontsize=title_font_size, va='top', ha='right')
    #axes[2].set_ylabel("FastSK AUC", size=axis_font_size)
    #axes[2].text(1.6, 1.1, '(c)', transform=axes[].transAxes, fontsize=title_font_size, va='top', ha='right')
    #axes[2].text(-0.1, 1.2, '(c)', transform=axes[0].transAxes, fontsize=title_font_size, va='top', ha='right')
    axes[2].set_title("NLP Test AUC", size=title_font_size)
    axes[2].tick_params(axis='both', which='major', labelsize=14)
    axes[2].legend(prop={'size': 12}, loc='lower right', edgecolor='black')

    fig.tight_layout()

    plt.savefig('auc_summary_fig.pdf')

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
    param_df = pd.read_csv('fastsk_summary_perfs_right_datasets.csv')
    g = int(param_df[param_df['Dataset'] == dataset]['g'])
    m = int(param_df[param_df['Dataset'] == dataset]['m'])
    max_I = int(special.comb(g, m))

    df = pd.read_csv(csv)

    I = list(df['I'])
    proportions = [i / max_I for i in I]
    acc = [100 * val for val in list(df['acc'])]
    auc = [100 * val for val in list(df['auc'])]
    
    fig, ax = plt.subplots()
    ax.plot(proportions, acc, label='Accuracy', marker='x', 
        color='b', linestyle='dashed')
    ax.plot(proportions, auc, label='AUC', marker='x', color='r')

    ax.set_xlabel(r'Proportion (and Number) of Iterations', size=axis_font_size)
    #ax.xaxis.set_major_locator(MultipleLocator(10))
    #ax.xaxis.set_minor_locator(MultipleLocator(5))
    ax.tick_params(axis='both', which='major', labelsize=14)
    #ax.set_yscale('log')
    ax.set_ylabel("Performance", size=axis_font_size)
    ax.set_ylim(bottom=90)
    #ax.set_xlim(left=0)
    x_pos = [0.1*i for i in range(6)]
    amounts = [int(max_I * p) for p in x_pos]
    labels = ['{:.1f} ({})'.format(pos, amount) for pos, amount in zip(x_pos, amounts)]
    ax.set_xticks(x_pos)
    ax.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=True,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=True
    )
    ax.set_xticklabels(labels, fontdict={'fontsize': 14})

    #ax.set_title('Max Iters ({})'.format(dataset))
    ax.legend(prop={'size': 15}, edgecolor='black')
    fig.tight_layout()

    outfile = dataset + "_I.pdf"
    print("Saving to {}".format(outfile))
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
            
            ax.plot(threads, exact, label='FastSK-Exact', color='b', marker='x')
            #ax.plot(threads, approx, label='FastSK (approx, 1 threa)', color='orange', marker='x')
            ax.plot(threads, i50, label='FastSK-Approx (1 thread)', color='g', marker='x')
            ax.plot(threads, gkm, label='gkmSVM-2.0', color='r', marker='o')
            #ax.plot(threads, i50, label=r'FastSK ($\leq$ 50 iterations)', color='magenta', marker='x')

            ax.set_xlabel("Number of Threads", size=axis_font_size)
            ax.xaxis.set_major_locator(MultipleLocator(2))
            ax.xaxis.set_minor_locator(MultipleLocator(1))
            ax.tick_params(axis='both', which='major')
            if c == 0:
                ax.set_ylabel("Time (s)", size=axis_font_size)
                ax.legend(prop={'size': 15}, edgecolor='black')
            ax.set_ylim(bottom=0)
            #ax.set_yscale('log')
            ax.set_title("{}".format(dataset))

            fig.tight_layout()
    
    plt.savefig(outfile)

def bigfig_I(csv_dir, outfile):
    param_df = pd.read_csv('fastsk_summary_perfs_right_datasets.csv')
    files = os.listdir(csv_dir)
    num_files = len(files)
    rows, cols = 4, 5
    threads = list(range(1, 21))

    fig, axes = plt.subplots(rows, cols)
    fig.set_size_inches(25, 15)

    dna_files = sorted([f for f in files if f.split('_')[0] in dna_datasets])
    prot_files = sorted([f for f in files if f.split('_')[0] in prot_datasets])
    files = dna_files + prot_files

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

            g = int(param_df[param_df['Dataset'] == dataset]['g'])
            m = int(param_df[param_df['Dataset'] == dataset]['m'])
            max_I = int(special.comb(g, m))

            if dataset in dna_datasets:
                dataset += ' (DNA)'
            else:
                dataset += ' (protein)'
            file = osp.join(csv_dir, filename)
            
            df = pd.read_csv(file)

            I = list(df['I'])
            proportions = [i / max_I for i in I]
            acc = [100 * val for val in list(df['acc'])]
            auc = [100 * val for val in list(df['auc'])]
            
            ax = axes[r][c]

            ax.plot(I, acc, label='Accuracy', marker='x', color='b')
            ax.plot(I, auc, label='AUC', marker='x', color='r')

            ax.set_xlabel(r'Proportion of Positions Sampled', size=axis_font_size)
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
            #labels = ['{} ({0:.1f})'.format(i, p) for i, p in zip(I, proportions)]
            #ax.set_xticklabels(labels, fontdict={'fontsize': 10})
            #sax.set_yscale('log')
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
    fig.set_size_inches(25, 7.5)

    dna_files = sorted([f for f in files if f.split('_')[0] in dna_datasets])
    prot_files = sorted([f for f in files if f.split('_')[0] in prot_datasets])
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

            # fastsk_exact = [v for v in list(df['FastSK-Exact']) if v != 0]
            # fastsk_approx_conv = [v for v in list(df['FastSK-Approx 1 thread']) if v != 0]
            # fastsk_approx_no_var = [v for v in list(df['FastSK-Approx 20 thread no variance 50 iters']) if v != 0]
            # gkm_exact = [v for v in list(df['gkm-Exact 20 thread']) if v != 0]
            # gkm_approx = [v for v in list(df['gkm-Approx 20 thread']) if v != 0]

            fastsk_approx_conv = [v for v in list(df['fastsk_I50']) if v != 0]
            gkm_exact = [v for v in list(df['gkm_exact']) if v != 0]
            #gkm_approx = [v for v in list(df['gkm-Approx 20 thread']) if v != 0]

            ax = axes[r][c]
            
            # ax.plot(g[:len(fastsk_exact)], fastsk_exact, 
            #     label='FastSK-Exact (20 threads)', 
            #     color='b', 
            #     marker='x')
            ax.plot(g[:len(fastsk_approx_conv)], fastsk_approx_conv, 
                label=r'FastSK', 
                color='b',
                marker='x')            
            # ax.plot(g[:len(fastsk_approx_no_var)], fastsk_approx_no_var, 
            #     label=r'FastSK-Approx (50 iterations)', 
            #     color='orange', 
            #     marker='x')
            ax.plot(g[:len(gkm_exact)], gkm_exact, 
                label='gkmSVM-2.0 (20 threads)', 
                color='r', 
                marker='o')            
            # ax.plot(g[:len(gkm_approx)], gkm_approx, 
            #     label='gkmSVM-2.0-Approx (20 threads)', 
            #     color='r', 
            #     marker='^')

            ax.set_xlabel(r'$g$', size=axis_font_size)
            ax.xaxis.set_major_locator(MultipleLocator(2))
            ax.xaxis.set_minor_locator(MultipleLocator(1))
            ax.tick_params(axis='both', which='major', labelsize=axis_font_size)
            if c == 0:
                ax.set_ylabel("Time (s)", size=30)
                ax.legend(prop={'size': 13}, edgecolor='black')
            #ax.set_ylim(bottom=0)
            ax.set_yscale('log')
            ax.set_title("{}".format(dataset), size=title_font_size)
    
    # fig.suptitle(r'Kernel Computation Time vs $g$ (Protein Data)',
    #     fontsize=24)
    # fig.subplots_adjust(top=0.88)
    fig.tight_layout()
    
    print("Saving figures to {}".format(outfile))
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


def bigfig_m(csv_dir, outfile):
    files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]

    dna_files = sorted([f for f in files if f.split('_')[0] in dna_datasets])
    prot_files = sorted([f for f in files if f.split('_')[0] in prot_datasets])
    
    #files = dna_files + prot_files
    files = prot_files

    num_files = len(files)
    rows, cols = 2, 5

    fig, axes = plt.subplots(rows, cols)
    #fig.set_size_inches(25, 15)
    fig.set_size_inches(25, 7.5)

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
            
            m = list(df['m'])

            fastsk_exact = [v for v in list(df['FastSK-Exact 20 thread']) if v != 0]
            fastsk_approx = [v for v in list(df['FastSK-Approx 1 thread']) if v != 0]
            fastsk_approx_no_var = [v for v in list(df['FastSK-Approx 1 thread no variance 50 iters']) if v != 0]
            gkm_exact = [v for v in list(df['gkmSVM-Exact 20 thread']) if v != 0]
            gkm_approx = [v for v in list(df['gkmSVM-Approx 20 thread']) if v != 0]
            
            ax = axes[r][c]

            # ax.plot(m[:len(fastsk_exact)], fastsk_exact,
            #     label=r'FastSK-Exact',
            #     marker='x', 
            #     color='b')
            
            # ax.plot(m[:len(fastsk_approx)], fastsk_approx,
            #     label=r'FastSK-Approx ($\leq gCk$ iterations)', 
            #     marker='x', 
            #     color='orange')

            ax.plot(m[:len(fastsk_approx_no_var)], fastsk_approx_no_var,
                label=r'FastSK', 
                marker='x', 
                color='blue')

            ax.plot(m[:len(gkm_exact)], gkm_exact,
                label=r'gkmSVM-2.0',
                marker='o',
                color='r')

            ax.plot(m[:4], gkm_approx[:4],
                label=r'gkmSVM-2.0-Approx ($m_{max} = 3$)',
                marker='^',
                color='r')

            ax.set_xlabel(r'$m$', size=30)
            ax.xaxis.set_major_locator(MultipleLocator(2))
            ax.xaxis.set_minor_locator(MultipleLocator(1))
            ax.tick_params(axis='both', which='major', labelsize=axis_font_size)
            if (c == 0):
                ax.set_ylabel("Time (s)", size=30)
                ax.legend(prop={'size': 10}, edgecolor='black')
            ax.set_yscale('log')
            ax.set_title(r'{}'.format(dataset), size=title_font_size)
            fig.tight_layout()
    
    print("Saving figures to {}".format(outfile))
    plt.savefig(outfile)

def big_fig_increase_g(csv_dir, outfile):
    files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]

    dna_files = sorted([f for f in files if f.split('_')[0] in dna_datasets])
    prot_files = sorted([f for f in files if f.split('_')[0] in prot_datasets])
    files = dna_files + prot_files
    num_files = len(files)
    rows, cols = 4, 5

    fig, axes = plt.subplots(rows, cols)
    fig.set_size_inches(25, 15)
    #fig.set_size_inches(25, 7.5)

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
            if dataset != 'ZZZ3':
                fastsk_approx_conv_auc = list(df['fastsk_approx_conv_auc'])
            gkm_approx_auc = list(df['gkm_approx_auc'])
            
            ax = axes[r][c]

            if dataset != 'ZZZ3':
                ax.plot(g, fastsk_approx_conv_auc,
                    label=r'FastSK',
                    marker='x', 
                    color='b')
            else:
                ax.plot(g, fastsk_approx_i50_auc,
                    label=r'FastSK',
                    marker='x', 
                    color='b')
            
            # ax.plot(g, fastsk_approx_i50_auc,
            #     label='FastSK', 
            #     marker='x', 
            #     color='b')

            ax.plot(g, gkm_approx_auc,
                label=r'gkmSVM-2.0-Approx',
                marker='^',
                color='r')

            ax.set_xlabel(r'$g$', size=30)
            ax.xaxis.set_major_locator(MultipleLocator(2))
            ax.xaxis.set_minor_locator(MultipleLocator(1))
            ax.tick_params(axis='both', which='major', labelsize=axis_font_size)
            if (c == 0):
                ax.set_ylabel("AUC", size=30)
                ax.legend(prop={'size': 12}, edgecolor='black')
            #ax.set_ylim(bottom=0, top=1)
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

def deep_learning_variance_plot(csv='variance_results.csv', outfile='variance_plot.pdf'):
    df = pd.read_csv(csv)
    num_samples = np.array(df['Num train samples'])
    fastsk_auc = np.array(df['fastsk_auc'])
    lstm_mean = np.array(df['lstm_mean'])
    lstm_lower_ci = np.array(df['lstm lower 95% CI'])
    lstm_upper_ci = np.array(df['lstm upper 95% CI'])
    charcnn_mean = np.array(df['charcnn_mean'])
    charcnn_lower_ci = np.array(df['charcnn lower 95% CI'])
    charcnn_upper_ci = np.array(df['charcnn upper 95% CI'])

    fig, ax = plt.subplots()
    ax.plot(num_samples, fastsk_auc, label='FastSK', marker='x', color='b')
    ax.plot(num_samples, charcnn_mean, label='CharCNN', marker='*', color='orange')
    ax.fill_between(num_samples, charcnn_lower_ci, charcnn_upper_ci, color='orange', alpha=0.5)
    ax.plot(num_samples, lstm_mean, label='LSTM', marker='o', color='r')
    ax.fill_between(num_samples, lstm_lower_ci, lstm_upper_ci, color='r', alpha=0.5)
    ax.set_xlabel(r'Number of Training Samples', size=axis_font_size)
    ax.xaxis.set_major_locator(MultipleLocator(2000))
    ax.xaxis.set_minor_locator(MultipleLocator(1000))
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_ylabel("Test AUC", size=axis_font_size)
    ax.set_ylim(bottom=0.50, top=1)
    ax.legend(prop={'size': 15}, loc='lower right')
    fig.tight_layout()
    plt.savefig(outfile)

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


def auc_vs_time_fig(d):
    # t vs g - g_times_nov14/CTCF_g_times_nov14.csv
    # auc vs g - nov18_g_auc/CTCF_nov18_g_auc.csv

    fastsk_approx_auc, fastsk_approx_time = [], []
    gkm_approx_auc, gkm_approx_time = [], []

    auc_g_csv = osp.join('nov18_g_auc', '{}_nov18_g_auc.csv'.format(d))
    time_g_csv = osp.join('g_times_nov14', '{}_g_times_nov14.csv'.format(d))
    auc_df = pd.read_csv(auc_g_csv)
    time_df = pd.read_csv(time_g_csv)

    fastsk_aucs = list(auc_df['fastsk_approx_auc'])
    fastsk_times = list(time_df['FastSK-Approx 1 thread no variance 50 iters'])
    gkm_aucs = list(auc_df['gkm_approx_auc'])
    gkm_times = list(time_df['gkm-Approx 20 thread'])

    fig, ax = plt.subplots()
    
    ax.scatter(fastsk_times, fastsk_aucs, 
        label='FastSK', color='b', marker='x')
    ax.scatter(gkm_times, gkm_aucs, 
        label='gkmSVM-Approx', color='r', marker='o')

    ax.set_xlabel('Kernel Computation Time (s)', size=axis_font_size)
    #ax.xaxis.set_major_locator(MultipleLocator(2))
    #ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.tick_params(axis='both', which='major')
    ax.set_ylabel('AUC', size=axis_font_size)
    #ax.set_ylim(bottom=0)
    #ax.set_xscale('log')
    ax.set_title("AUC vs Kernel Computation Time ({})".format(d))
    ax.legend(prop={'size': 13}, edgecolor='black')

    fig.tight_layout()
    outfile = '{}_auc_vs_time.pdf'.format(d)
    plt.savefig(outfile)

    print("Saving figure to {}".format(outfile))

## This is figure 7 from the Jan 30 submission
def fig7(approx_exact_csv, auc_time_csv):
    ## left subfig: approx vs exact
    df = pd.read_csv(approx_exact_csv)
    exact_auc = list(df['exact_auc'])
    i50_auc = list(df['I50_auc'])
    
    # convergence 
    # avg_auc_loss = 100 * sum([a - e for a, e in zip(approx_auc, exact_auc)]) / len(exact_auc)
    # avg_acc_loss = sum([a - e for a, e in zip(approx_acc, exact_acc)]) / len(exact_acc)
    # print("Convergence: AUC loss: {}%, Acc loss: {}%".format(avg_auc_loss, avg_acc_loss))

    # # 50 iters
    # avg_auc_loss = 100 * sum([a - e for a, e in zip(i50_auc, exact_auc)]) / len(exact_auc)
    # avg_acc_loss = sum([a - e for a, e in zip(i50_acc, exact_acc)]) / len(exact_acc)
    # print("50 iters: AUC loss: {}%, Acc loss: {}%".format(avg_auc_loss, avg_acc_loss))

    # parity line
    parity = np.linspace(0, 1)

    # approx vs exact auc line plot
    fig, axes = plt.subplots(1, 3)
    fig.set_size_inches(17, 5)

    axes[0].scatter(exact_auc, i50_auc, color='b', marker='x',zorder=2)
    axes[0].plot(parity, parity, color='r', zorder=1)
    axes[0].set_xlabel("FastSK-Exact AUC", size=axis_font_size)
    axes[0].set_ylabel("FastSK-Approx AUC", size=axis_font_size)
    axes[0].set_title("(a) FastSK-Approx vs FastSK-Exact AUC", size=15)
    axes[0].tick_params(axis='both', which='major', labelsize=14)

    ## center subfig: auc vs time
    fastsk_approx_auc, fastsk_approx_time = [], []
    gkm_approx_auc, gkm_approx_time = [], []

    auc_g_csv = osp.join('nov18_g_auc', '{}_nov18_g_auc.csv'.format('EP300'))
    time_g_csv = osp.join('g_times_nov14', '{}_g_times_nov14.csv'.format('EP300'))
    auc_df = pd.read_csv(auc_g_csv)
    time_df = pd.read_csv(time_g_csv)

    fastsk_aucs = list(auc_df['fastsk_approx_auc'])
    fastsk_times = list(time_df['FastSK-Approx 1 thread no variance 50 iters'])
    gkm_aucs = list(auc_df['gkm_approx_auc'])
    gkm_times = list(time_df['gkm-Approx 20 thread'])
    
    axes[1].scatter(fastsk_times, fastsk_aucs, label='FastSK', color='b', marker='x')
    axes[1].scatter(gkm_times, gkm_aucs, label='gkmSVM-Approx', color='r', marker='o')

    axes[1].set_xlabel('Kernel Computation Time (s)', size=axis_font_size)
    #ax.xaxis.set_major_locator(MultipleLocator(2))
    #ax.xaxis.set_minor_locator(MultipleLocator(1))
    axes[1].tick_params(axis='both', which='major')
    axes[1].set_ylabel('AUC', size=axis_font_size)
    axes[1].tick_params(axis='both', which='minor', labelsize=axis_font_size)
    #ax.set_ylim(bottom=0)
    #ax.set_xscale('log')
    axes[1].set_title("(b) AUC vs Time ({})".format('EP300'), size=15)
    axes[1].legend(prop={'size': 13}, edgecolor='black', loc='lower left')

    csv = 'dec15_g_auc_results/EP300_dec15_g_auc.csv'
    df = pd.read_csv(csv)
    g = list(df['g'])

    fastsk_approx_i50_auc = list(df['fastsk_approx_i50_auc'])
    gkm_approx_auc = list(df['gkm_approx_auc'])

    axes[2].plot(g, fastsk_approx_i50_auc, label=r'FastSK', marker='x', color='b')
    axes[2].plot(g, gkm_approx_auc, label=r'gkmSVM-2.0-Approx', marker='^', color='r')
    axes[2].set_xlabel(r'$g$', size=axis_font_size)
    axes[2].xaxis.set_major_locator(MultipleLocator(2))
    axes[2].xaxis.set_minor_locator(MultipleLocator(1))
    axes[2].tick_params(axis='both', which='major', labelsize=axis_font_size)
    axes[2].set_ylabel("AUC", size=15)
    axes[2].legend(prop={'size': 12}, edgecolor='black')
    #ax.set_ylim(bottom=0, top=1)
    axes[2].set_title(r'(c) AUC vs g (EP300)', size=15)


    fig.tight_layout()
    outfile = 'approx_analysis.pdf'
    plt.savefig(outfile)

    print("Saving figure to {}".format(outfile))

## version 1: speedup summary barchart averaged over varying g
def speedup_barchart_avg_g(type_, outfile='speedup_barchart_dna.pdf'):
    assert type_ in ['dna', 'protein']
    datasets = dna_datasets if type_ == 'dna' else prot_datasets
    csv_dir = './g_time' if type_ == 'dna' else './dec14_g_times'
    avgs, maxes, names = [], [], []
    for i, d in enumerate(datasets):
        csv = osp.join(csv_dir, d + '_g_time.csv')
        df = pd.read_csv(csv)
        fastsk_approx = list(df['fastsk_I50']) if type_ == 'dna' else list(df['FastSK-Approx 1 thread'])
        if type_ == 'dna':
            gkm_exact = [v for v in list(df['gkm_exact']) if v != 0]
        else:
            gkm_exact = [v for v in list(df['gkm-Exact 20 thread']) if v != 0]
        gkm_approx = list(df['gkm_approx'])
        avg = [gkm / fastsk for (gkm, fastsk) in zip(gkm_exact, fastsk_approx[:len(gkm_exact)])]
        avg = sum(avg) / len(avg)
        max_ = max([gkm / fastsk for (gkm, fastsk) in zip(gkm_exact, fastsk_approx[:len(gkm_exact)])])
        avgs.append(avg)
        maxes.append(max_)
        if d == 'EP300_47848':
            d = 'EP300 (2)'
        names.append(d)

    #avgs, names = zip(*sorted(zip(avgs, names)))

    x_pos = [i*2 for i, _ in enumerate(dna_datasets)]

    fig, ax = plt.subplots(1,2)
    fig.set_size_inches(15, 5)
    subplot_names = ["Average", "Max"]
    y, labels = zip(*sorted(zip(avgs, names)))
    ax[0].bar(x_pos, y, edgecolor='black', color='silver', hatch='//')
    ax[0].set_xlabel('Dataset', size=axis_font_size)
    ax[0].set_ylabel('Average Speedup', size=axis_font_size)
    #ax[0].set_title('Average Speedup', size=12)
    ax[0].set_xticks(x_pos)
    ax[0].tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=True,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=True
    )
    ax[0].set_xticklabels(labels, fontdict={'fontsize': 10})
    for i, d in enumerate(y):
        ax[0].text(x_pos[i], y[i] + 2, str('{0:.1f}x'.format(d)), fontsize=8, ha='center', fontweight='bold')
    ax[0].text(0, 1.1, '(a)', transform=ax[0].transAxes, fontsize=title_font_size, va='top', ha='right')

    y, labels = zip(*sorted(zip(maxes, names)))
    ax[1].bar(x_pos, y, edgecolor='black', color='silver', hatch='//')
    ax[1].set_xlabel('Dataset', size=axis_font_size)
    ax[1].set_ylabel('Max Speedup', size=axis_font_size)
    #ax[1].set_title('Max Speedup', size=12)
    ax[1].set_xticks(x_pos)
    ax[1].tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=True,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=True
    )
    ax[1].set_xticklabels(labels, fontdict={'fontsize': 10})
    for i, d in enumerate(y):
        ax[1].text(x_pos[i], y[i] + 10, str('{0:.1f}x'.format(d)), fontsize=8, ha='center', fontweight='bold')
    ax[1].text(0, 1.1, '(b)', transform=ax[1].transAxes, fontsize=title_font_size, va='top', ha='right')
    

    fig.tight_layout()
    plt.savefig(outfile)

def dna_and_prot_convergence_fig(dna_csv, prot_csv, outfile):
    '''Figure 13 in arXiv version. Shows DNA iters/convergence on the
    left and prot iters/convergence on the right.
    '''

    fig, ax = plt.subplots(1, 2)
    fig.set_size_inches(15, 5)

    for i in range(0, 2):
        type_ = 'DNA' if i == 0 else 'Protein'
        if type_ == 'DNA':
            df = pd.read_csv(dna_csv)
        else:
            df = pd.read_csv(prot_csv)

        dataset = list(df['dataset'])[0]
        g, m = list(df['g'])[0], list(df['m'])[0]
        max_I = int(special.comb(g, m))
        iters = list(df['iters'])
        iters = [100 * i / max_I for i in iters]

        mean_auc = list(df['mean auc'])
        lower_auc = list(df['lower auc'])
        upper_auc = list(df['upper auc'])

        mean_stdev = list(df['mean stdev'])
        lower_stdev = list(df['lower stdev'])
        upper_stdev = list(df['upper stdev'])

        auc_handle = ax[i].plot(iters, mean_auc, label='AUC', marker='x', color='r')
        ax[i].fill_between(iters, lower_auc, upper_auc, color='r', alpha=0.5)
        
        ax2 = ax[i].twinx()
        sd_handle = ax2.plot(iters, mean_stdev, label='Kernel variance', marker='o', color='orange')
        ax2.fill_between(iters, lower_stdev, upper_stdev, color='orange', alpha=0.5)
        ax2.grid(None)
        ax2.tick_params(axis='both', which='major', labelsize=14)
        if i == 1:
            ax2.set_ylabel("Kernel Variance", size=axis_font_size, rotation=-90, labelpad=20)
        ax2.set_ylim(bottom=0, top=1)

        ax[i].set_xlabel(r'Percent (and Number) of Iterations', size=axis_font_size)

        x_pos = [10 * i for i in range(0, 10, 2)]
        amounts = [int(max_I * p / 100) for p in x_pos]
        labels = ['{:.1f}% ({})'.format(pos, amount) for pos, amount in zip(x_pos, amounts)]
        ax[i].set_xticks(x_pos)
        ax[i].tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=True,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=True
        )
        ax[i].set_xticklabels(labels, fontdict={'fontsize': 12})
        ax[i].tick_params(axis='both', which='major', labelsize=12)
        ax2.tick_params(axis='both', which='major', labelsize=12)

        if i == 0:
            ax[i].set_ylabel("AUC", size=axis_font_size)
            ax[i].text(0, 1.1, '(a)', transform=ax[i].transAxes, fontsize=title_font_size, va='top', ha='right')
        else:
            ax[i].text(0, 1.1, '(b)', transform=ax[i].transAxes, fontsize=title_font_size, va='top', ha='right')
    
        #ax[i].tick_params(axis='both', which='major', labelsize=14)
        ax[i].set_ylim(bottom=0.50, top=1)
    
        lines = auc_handle + sd_handle
        labels = [l.get_label() for l in lines]
        ax[i].legend(lines, labels, prop={'size': 10}, loc='lower right')

        ax[i].set_title("Performance vs Iterations - {} ({})".format(dataset, type_))
    
    fig.tight_layout()
    
    print("Saving figure to ", outfile)
    plt.savefig(outfile)


def stdev_vs_I(csv, outfile):
    '''Show accuracy, AUC, and kernel matrix variance as functions
    of the number of approximation algorithm iterations.
    In the paper, we show this for EP300 (figure 2).
    '''
    df = pd.read_csv(csv)
    dataset = list(df['dataset'])[0]
    g, m = list(df['g'])[0], list(df['m'])[0]
    max_I = int(special.comb(g, m))
    iters = list(df['iters'])
    mean_acc = list(df['mean acc'])
    lower_acc = list(df['lower acc'])
    upper_acc = list(df['upper acc'])

    mean_auc = list(df['mean auc'])
    lower_auc = list(df['lower auc'])
    upper_auc = list(df['upper auc'])

    mean_stdev = list(df['mean stdev'])
    lower_stdev = list(df['lower stdev'])
    upper_stdev = list(df['upper stdev'])

    iters = [100 * i / max_I for i in iters]

    fig, ax = plt.subplots()
    acc_handle = ax.plot(iters, mean_acc, label='Accuracy', marker='x', color='b')
    ax.fill_between(iters, lower_acc, upper_acc, color='b', alpha=0.5)
    auc_handle = ax.plot(iters, mean_auc, label='AUC', marker='o', color='r')
    ax.fill_between(iters, lower_auc, upper_auc, color='r', alpha=0.5)

    ax2 = ax.twinx()
    sd_handle = ax2.plot(iters, mean_stdev, label='Kernel variance', marker='o', color='orange')
    ax2.fill_between(iters, lower_stdev, upper_stdev, color='orange', alpha=0.5)
    ax2.grid(None)
    ax2.tick_params(axis='both', which='major', labelsize=14)
    ax2.set_ylabel("Kernel Variance", size=axis_font_size, rotation=-90, labelpad=20)
    ax2.set_ylim(bottom=0, top=1)

    ax.set_xlabel(r'Number of Iterations', size=axis_font_size)
    ax.set_ylabel("Performance", size=axis_font_size)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_ylim(bottom=0.50, top=1)

    x_pos = [10 * i for i in range(0, 11, 2)]
    amounts = [int(max_I * p / 100) for p in x_pos]
    labels = ['{}\n{:.1f}%'.format(amount, pos) for pos, amount in zip(x_pos, amounts)]
    ax.set_xticks(x_pos)
    ax.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=True,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=True
    )
    ax.set_xticklabels(labels, fontdict={'fontsize': 12})
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax2.tick_params(axis='both', which='major', labelsize=12)
    
    lines = acc_handle + auc_handle + sd_handle
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, prop={'size': 12}, loc='lower right')

    #ax.set_title("Performance vs Iterations ({})".format(dataset))
    fig.tight_layout()
    plt.savefig(outfile)

def bigfig_stdev_I(csv_dir, outfile):
    files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]

    dna_files = sorted([f for f in files if f.split('_')[0] in dna_datasets])
    prot_files = sorted([f for f in files if f.split('_')[0] in prot_datasets])
    nlp_files = sorted([f for f in files if f.split('_')[0] in nlp_datasets])

    files = dna_files + prot_files

    num_files = len(files)
    rows, cols = 4, 5

    fig, axes = plt.subplots(rows, cols)
    fig.set_size_inches(25, 15)

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

            dataset = list(df['dataset'])[0]
            g, m = list(df['g'])[0], list(df['m'])[0]
            max_I = int(special.comb(g, m))
            iters = list(df['iters'])
            mean_acc = list(df['mean acc'])
            lower_acc = list(df['lower acc'])
            upper_acc = list(df['upper acc'])

            mean_auc = list(df['mean auc'])
            lower_auc = list(df['lower auc'])
            upper_auc = list(df['upper auc'])

            mean_stdev = list(df['mean stdev'])
            lower_stdev = list(df['lower stdev'])
            upper_stdev = list(df['upper stdev'])            

            ax = axes[r][c]

            acc_handle = ax.plot(iters, mean_acc, label='Accuracy', marker='x', color='b')
            ax.fill_between(iters, lower_acc, upper_acc, color='b', alpha=0.5)
            auc_handle = ax.plot(iters, mean_auc, label='AUC', marker='o', color='r')
            ax.fill_between(iters, lower_auc, upper_auc, color='r', alpha=0.5)

            ax.tick_params(
                axis='x',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom=True,      # ticks along the bottom edge are off
                top=False,         # ticks along the top edge are off
                labelbottom=True
            )

            ax2 = ax.twinx()
            sd_handle = ax2.plot(iters, mean_stdev, label='Kernel variance', marker='o', color='orange')
            ax2.fill_between(iters, lower_stdev, upper_stdev, color='orange', alpha=0.5)
            ax2.grid(None)
            #ax2.tick_params(axis='both', which='major', labelsize=14)
            if dataset == 'Pbde':
                ax2.set_ylim(bottom=0, top=100)
            elif dataset == 'TP53':
                ax2.set_ylim(bottom=0, top=10)
            else:
                ax2.set_ylim(bottom=0, top=1)

            ax.set_xlabel(r'Number of Iterations', size=axis_font_size)

            ax.set_ylim(bottom=0.50, top=1)
            ax.set_title(r'{}'.format(dataset), size=title_font_size)

            if (c == 0):
                lines = acc_handle + auc_handle + sd_handle
                labels = [l.get_label() for l in lines]
                ax.legend(lines, labels, prop={'size': 10}, loc='lower right')
                ax.set_ylabel("Performance", size=axis_font_size)
                ax2.set_ylabel("Kernel Variance", size=axis_font_size, rotation=-90, labelpad=20)
            
            fig.tight_layout()
    
    print("Saving figures to {}".format(outfile))
    plt.savefig(outfile)


bigfig_stdev_I('stdevs', 'bigfig_I_stdevs.pdf')
#dna_and_prot_convergence_fig('stdevs/EP300_stdev_auc_iters.csv', 'stdevs/1.1_stdev_auc_iters.csv', 'dna_prot_iters_stdev.pdf')

#stdev_vs_I('stdevs/EP300_stdev_auc_iters.csv', 'EP300_stdev_auc_iters.pdf')

#fig7('fastsk_summary_perfs_right_datasets.csv', None)

#auc_vs_time_fig(d='EP300')

#bigfig_m('jan_2020_vary_m', 'vary_m_protein.pdf')

#approx_vs_exact('fastsk_summary_perfs_right_datasets.csv', 'i50')
#nlp_perfs('nlp_results.csv')
#big_fig_threads('./vary_thread_results', 'bigfig_threads_I50.pdf')
#vary_g('EP300_vary_g.csv', 'EP300')
#bigfig_g_time('./dec14_g_times/', 'deleteme.png')
#bigfig_g_time('./g_time/', 'g_time_dna.pdf')
#bigfig_g_time('./dec14_g_times/', 'deleteme.png')
#big_fig_increase_g('./dec15_g_auc_results', 'bigfig_g_auc_results.pdf')
#get_speeds_over_g('./g_time')
#vary_delta('1.1_vary_delta.csv', '1.1')
#bigfig_I('./vary_I', 'vary_I_proportions.pdf')
#vary_I('./vary_I/EP300_vary_I.csv', 'EP300')
#big_fig_delta('./vary_delta_csvs', 'bigfig_delta.pdf')
#big_fig_increase_g('./increase_g', 'bigfig_increase_g.pdf')
#big_fig_increase_g('./increase_g_k6', 'bigfig_increase_g_k6.pdf')
#big_fig_increase_g('./k6C', 'k6C.pdf')
#big_fig_increase_g('./g_auc_k6', 'g_auc_k6.pdf')
#big_fig_increase_g('./increase_g_k4', 'protein_auc_k4.pdf')
#vary_num_threads('./vary_thread_results/RAD21_vary_threads.csv', 'RAD21')
#increase_g('./increase_g_k6/RAD21_increase_g_k6.csv', 'RAD21')
#increase_g_time('./g_time_k6/CTCF_vary_g.csv', 'CTCF')
#increase_g_time('./g_time/CTCF_g_time.csv', 'CTCF')
#lstm_proportions('lstm_proportions2.csv', 'lstm_props.pdf')

#auc_summary_fig()
#deep_learning_variance_plot()
#speedup_barchart_avg_g(type_='dna')
