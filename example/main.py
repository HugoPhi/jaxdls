from plugins.lrkit.executer import KFlodCrossExecuter

from clfs import lstm, gru
from data_process import X_train, X_test, y_train, y_test

excr = KFlodCrossExecuter(
    X_train, y_train, X_test, y_test,
    clf_dict={
        'gru': gru(lr=0.01, epoches=2, batch_size=64),
        'lstm': lstm(lr=0.01, epoches=2, batch_size=64),
    },
    n_bootstraps=10,
    metric_list=['accuracy', 'macro_f1', 'micro_f1', 'avg_recall'],
    log=False,
    log_dir='./log/',
)

excr.run_all()
