python3 -u src/main_incremental.py --gpu 0 --results-path exp1_1 --eval-on-train --batch-size 128 --wandb-project exp1_1_lwm --nepoch 200 --approach lwm --dkd-control deterministic --dkd-m 0.5 --dkd-switch end --dkd-shape one --distill-percent 0.8 --ikr-control none



python3 -u src/main_incremental.py --gpu 0 --results-path exp1_1 --eval-on-train --batch-size 128 --wandb-project exp1_1_lwm --nepoch 200 --approach lwm --dkd-control deterministic --dkd-m 1.0 --dkd-switch mid --dkd-shape one --distill-percent 0.6 --ikr-control none
python3 -u src/main_incremental.py --gpu 0 --results-path exp1_1 --eval-on-train --batch-size 128 --wandb-project exp1_1_lwm --nepoch 200 --approach lwm --dkd-control deterministic --dkd-m 1.0 --dkd-switch end --dkd-shape one --distill-percent 0.6 --ikr-control none
python3 -u src/main_incremental.py --gpu 0 --results-path exp1_1 --eval-on-train --batch-size 128 --wandb-project exp1_1_lwm --nepoch 200 --approach lwm --dkd-control deterministic --dkd-m 1.0 --dkd-switch first_end --dkd-shape one --distill-percent 0.6 --ikr-control none
python3 -u src/main_incremental.py --gpu 0 --results-path exp1_1 --eval-on-train --batch-size 128 --wandb-project exp1_1_lwm --nepoch 200 --approach lwm --dkd-control deterministic --dkd-m 1.0 --dkd-switch ten_to_ten --dkd-shape one --distill-percent 0.6 --ikr-control none
## 40%KD
python3 -u src/main_incremental.py --gpu 0 --results-path exp1_1 --eval-on-train --batch-size 128 --wandb-project exp1_1_lwm --nepoch 200 --approach lwm --dkd-control deterministic --dkd-m 1.0 --dkd-switch cycle --dkd-shape one --distill-percent 0.4 --ikr-control none
python3 -u src/main_incremental.py --gpu 0 --results-path exp1_1 --eval-on-train --batch-size 128 --wandb-project exp1_1_lwm --nepoch 200 --approach lwm --dkd-control deterministic --dkd-m 1.0 --dkd-switch first --dkd-shape one --distill-percent 0.4 --ikr-control none