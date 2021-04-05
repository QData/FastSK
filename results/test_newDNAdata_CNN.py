import os.path as osp
import subprocess

# from cnn_hyperTrTune import hyper

datasets = [
    "BroadGM12878Ctcf",
    "HaibA549Foxa1V0416102Dex100nm",
    "HaibGM12878Cebpbsc150V0422111",
    "SydhGm12878MaxIggmus",
    "SydhHepg2Hnf4aForskIn",
    "SydhGm12878Jund",
    "SydhGm12878Yy1",
    "SydhK562Gata1Ucd",
    "db.GATA1_K562_GATA-1_USC",
]

for dataset in datasets:
    train_file = osp.join("../data/encode/", dataset + ".train.fasta")
    print("train_file = ", train_file)
    test_file = osp.join("../data/encode/", dataset + ".test.fasta")
    print("test_file = ", test_file)

    log_dir = osp.join("testWrapperResult/", "{}_cnn_results".format(dataset))

    for trn_size in [1.0, 0.8, 0.6, 0.4, 0.2]:
        for opt in ["sgd", "adam"]:
            for lr in [1e-2, 8e-3]:
                # hyper(opt, lr, trn_size, train_file, test_file, dataset)
                command = [
                    "python",
                    "../../results/neural_nets/cnn_hyperTrTune.py",
                    "--trn",
                    str(train_file),
                    "--tst",
                    str(test_file),
                    "--trn_size",
                    str(trn_size),
                    "--lr",
                    str(lr),
                    "--datasetTag",
                    str(dataset),
                    "--log_dir",
                    log_dir,
                    "--opt_mtd",
                    str(opt),
                    "--epochs",
                    str(200),
                ]
                print(" ".join(command))
                output = subprocess.check_output(command)
