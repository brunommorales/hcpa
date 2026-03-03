|abordagem|melhor_metrica|pior_metrica|resumo|
|---|---|---|---|
|PyTorch Base|AUC = 0.918|SPEC = 0.504|AUC 0.918; Spec 0.504; Thr 441 img/s; Mem 18868 MB; Train 6292s|
|TensorFlow Base|AUC = 0.986|THROUGHPUT = 682.888 img/s|AUC 0.986; Spec 0.947; Thr 683 img/s; Mem 11811 MB; Train 3707s|
|MONAI Base|TTA = 17.877 s|THROUGHPUT = 946.416 img/s|AUC 0.971; Spec 0.864; Thr 946 img/s; Mem 7341 MB; Train 2969s|
|PyTorch Opt|SPEC = 0.944|TTA = 272.554 s|AUC 0.985; Spec 0.944; Thr 1374 img/s; Mem 15204 MB; Train 2296s|
|TensorFlow Opt|THROUGHPUT = 1977.836 img/s|AUC = 0.956|AUC 0.956; Spec 0.777; Thr 1978 img/s; Mem 3712 MB; Train 2329s|
|MONAI Opt|TRAIN_TIME = 1941.510 s|THROUGHPUT = 1379.454 img/s|AUC 0.976; Spec 0.932; Thr 1379 img/s; Mem 6800 MB; Train 1942s|