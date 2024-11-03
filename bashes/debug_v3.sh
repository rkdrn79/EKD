# lwf 테스트
## --results-path exp1_3

## 100%KD ada_switch_1 _ one // m = 2.0
python3 -u src/main_incremental.py --gpu 0 --results-path exp1_3 --eval-on-train --batch-size 128 --wandb-project exp1_3_lwf --nepoch 200 --approach lwf --dkd-control adaptive --dkd-m 2.0 --dkd-switch ada_switch_1 --dkd-shape one --distill-percent 1 --ikr-control none
## 20%KD ada_switch_2 _ one
python3 -u src/main_incremental.py --gpu 0 --results-path exp1_3 --eval-on-train --batch-size 128 --wandb-project exp1_3_lwf --nepoch 200 --approach lwf --dkd-control adaptive --dkd-m 2.0 --dkd-switch ada_switch_2 --dkd-shape one --distill-percent 0.2 --ikr-control none
## 20%KD ada_switch_3 _ one
python3 -u src/main_incremental.py --gpu 0 --results-path exp1_3 --eval-on-train --batch-size 128 --wandb-project exp1_3_lwf --nepoch 200 --approach lwf --dkd-control adaptive --dkd-m 2.0 --dkd-switch ada_switch_3 --dkd-shape one --distill-percent 0.2 --ikr-control none
## 20%KD ada_switch_4 _ one
python3 -u src/main_incremental.py --gpu 0 --results-path exp1_3 --eval-on-train --batch-size 128 --wandb-project exp1_3_lwf --nepoch 200 --approach lwf --dkd-control adaptive --dkd-m 2.0 --dkd-switch ada_switch_4 --dkd-shape one --distill-percent 0.2 --ikr-control none
## 20%KD ada_switch_5 _ one
python3 -u src/main_incremental.py --gpu 0 --results-path exp1_3 --eval-on-train --batch-size 128 --wandb-project exp1_3_lwf --nepoch 200 --approach lwf --dkd-control adaptive --dkd-m 2.0 --dkd-switch ada_switch_5 --dkd-shape one --distill-percent 0.2 --ikr-control none
