Dieses Repository beinhaltet eigenversuche zur Implementierung von Branch and Bound/Cut sowie eine weiterentwicklung des Frameworks von Konrad Adamski

Eigenversuche beinhaltet schrittweise änderungen und die evolution der eigenen Branch and Bound implmentierung
BnB4.py ist so das beste normale Ansatz ohne zusätze
BnB_copy_3.py ist mit einer strategie nach startzeit zu sortieren
framework_final.py ist mit dem pyscipopt framework

und Adamskis Framework beinhaltet extra Datein, welchen den Branch and Bound eingliedern
darunter vor allem
- run_single_bnb_experiment.py
- src/BnB_Experiment_Runner.py
- src/solvers/heuristics/BNB_Scheduler.py
