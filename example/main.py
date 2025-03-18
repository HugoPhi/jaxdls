from plugins.lrkit.executer import NonValidExecuter

from clfs import (
    lstm, gru,
    bilstm, bigru,
    conv1dx3
)

from data_process import X_train, X_test, y_train, y_test

excr = NonValidExecuter(
    X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
    clf_dict={
        'gru': gru(lr=0.005, epoches=50, batch_size=64),
        'lstm': lstm(lr=0.005, epoches=40, batch_size=64),
        'bigru': bigru(lr=0.005, epoches=50, batch_size=64),
        'bilstm': bilstm(lr=0.005, epoches=100, batch_size=64),
        'conv1dx3': conv1dx3(lr=0.001, epoches=100, batch_size=128),
    },
    metric_list=['accuracy', 'macro_f1', 'micro_f1', 'avg_recall'],
    log=True,
    log_dir='./log/',
)

excr.run_all(time=True)
# excr.run('gru')
# excr.format_print(time=True)
