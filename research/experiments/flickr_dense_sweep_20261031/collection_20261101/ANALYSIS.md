# Flickr Dense sweep: validation-only analysis

32/32 output inventories/checkpoints/splits and scheduler DONE identities verified.
Single seed101; architecture exploration is not independent-seed confirmation.
Test labels were absent from training inputs; no held-out test outcome is used.
Recommendation is provisional: finalize architecture selection before any later test evaluation.

| Model | Layers | Width | Validation accuracy | Validation macro-F1 | Epochs | Stop | Actual host | Train seconds |
|---|---:|---:|---:|---:|---:|---|---|---:|
|gcn|2|32|0.520000|0.210914|500|max_epochs|polus-c3-ib.bmc.hpc.cs.msu.ru|571.81|
|gcn|2|64|0.524090|0.216653|500|max_epochs|polus-c2-ib.bmc.hpc.cs.msu.ru|621.57|
|gcn|2|128|0.522689|0.220721|500|max_epochs|polus-c4-ib.bmc.hpc.cs.msu.ru|634.03|
|gcn|2|256|0.523305|0.222674|500|max_epochs|polus-c4-ib.bmc.hpc.cs.msu.ru|993.26|
|gcn|3|32|0.507283|0.195702|500|max_epochs|polus-c3-ib.bmc.hpc.cs.msu.ru|844.06|
|gcn|3|64|0.510532|0.196887|500|max_epochs|polus-c2-ib.bmc.hpc.cs.msu.ru|987.71|
|gcn|3|128|0.510588|0.202871|500|max_epochs|polus-c4-ib.bmc.hpc.cs.msu.ru|1081.00|
|gcn|3|256|0.515574|0.211909|500|max_epochs|polus-c4-ib.bmc.hpc.cs.msu.ru|1805.22|
|gcn|4|32|0.515294|0.206699|500|max_epochs|polus-c3-ib.bmc.hpc.cs.msu.ru|1123.69|
|gcn|4|64|0.519888|0.213214|500|max_epochs|polus-c2-ib.bmc.hpc.cs.msu.ru|1362.80|
|gcn|4|128|0.524874|0.219104|500|max_epochs|polus-c4-ib.bmc.hpc.cs.msu.ru|1518.10|
|gcn|4|256|0.523529|0.218134|500|max_epochs|polus-c4-ib.bmc.hpc.cs.msu.ru|2614.63|
|gcn|5|32|0.504034|0.191607|500|max_epochs|polus-c4-ib.bmc.hpc.cs.msu.ru|815.14|
|gcn|5|64|0.510868|0.202174|500|max_epochs|polus-c3-ib.bmc.hpc.cs.msu.ru|2056.46|
|gcn|5|128|0.510084|0.205134|500|max_epochs|polus-c4-ib.bmc.hpc.cs.msu.ru|1968.40|
|gcn|5|256|0.511821|0.202793|500|max_epochs|polus-c4-ib.bmc.hpc.cs.msu.ru|3434.89|
|graphsage|2|32|0.525154|0.198156|500|max_epochs|polus-c2-ib.bmc.hpc.cs.msu.ru|765.38|
|graphsage|2|64|0.525826|0.209716|500|max_epochs|polus-c2-ib.bmc.hpc.cs.msu.ru|938.00|
|graphsage|2|128|0.526331|0.215219|446|early_stopping_patience|polus-c2-ib.bmc.hpc.cs.msu.ru|1188.29|
|graphsage|2|256|0.528403|0.214996|390|early_stopping_patience|polus-c4-ib.bmc.hpc.cs.msu.ru|1089.11|
|graphsage|3|32|0.529132|0.206457|500|max_epochs|polus-c3-ib.bmc.hpc.cs.msu.ru|1153.56|
|graphsage|3|64|0.536863|0.222532|500|max_epochs|polus-c2-ib.bmc.hpc.cs.msu.ru|1296.23|
|graphsage|3|128|0.538824|0.236442|473|early_stopping_patience|polus-c3-ib.bmc.hpc.cs.msu.ru|2188.93|
|graphsage|3|256|0.538655|0.234963|379|early_stopping_patience|polus-c4-ib.bmc.hpc.cs.msu.ru|1731.17|
|graphsage|4|32|0.528627|0.207484|500|max_epochs|polus-c2-ib.bmc.hpc.cs.msu.ru|1199.46|
|graphsage|4|64|0.540168|0.231891|500|max_epochs|polus-c2-ib.bmc.hpc.cs.msu.ru|1709.57|
|graphsage|4|128|0.541513|0.236311|487|early_stopping_patience|polus-c4-ib.bmc.hpc.cs.msu.ru|1863.19|
|graphsage|4|256|0.540280|0.240890|408|early_stopping_patience|polus-c4-ib.bmc.hpc.cs.msu.ru|2629.71|
|graphsage|5|32|0.525602|0.207207|500|max_epochs|polus-c3-ib.bmc.hpc.cs.msu.ru|1679.95|
|graphsage|5|64|0.537759|0.224721|500|max_epochs|polus-c2-ib.bmc.hpc.cs.msu.ru|2035.76|
|graphsage|5|128|0.540336|0.227629|500|max_epochs|polus-c4-ib.bmc.hpc.cs.msu.ru|2349.54|
|graphsage|5|256|0.541457|0.232325|500|max_epochs|polus-c4-ib.bmc.hpc.cs.msu.ru|4085.74|

Training cost comparisons must be conditioned on actual execution host/threads.
No global cross-host architecture speed ranking or inference acceleration claim is made.
Max-epoch stops, latest-best checkpoints, loss excursions and train/validation gaps require curve review.
Finishing patience is not proof of generalization or global optimization.
Depth2/4/5 finalists are not yet compatible with the unchanged three-layer V2 pruning controllers.
The at-most-two/model recommendations require seeds202/303 confirmation; no such jobs are started.
