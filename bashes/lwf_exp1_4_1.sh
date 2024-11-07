
## adaptive

# 20%KD ada_switch_2 _ one
python3 -u src/main_incremental.py --gpu 1 --results-path exp1_4 --eval-on-train --batch-size 128 --wandb-project exp1_4_lwf --nepoch 200 --approach lwf --dkd-control adaptive --dkd-m 2.0 --dkd-switch ada_switch_2 --dkd-shape one --distill-percent 0.2 --ikr-control deterministic --ikr-m 2.0 --ikr-switch all --pk-approach average
# 20%KD ada_switch_3 _ one
python3 -u src/main_incremental.py --gpu 1 --results-path exp1_4 --eval-on-train --batch-size 128 --wandb-project exp1_4_lwf --nepoch 200 --approach lwf --dkd-control adaptive --dkd-m 2.0 --dkd-switch ada_switch_3 --dkd-shape one --distill-percent 0.2 --ikr-control deterministic --ikr-m 2.0 --ikr-switch all --pk-approach average
# 20%KD ada_switch_4 _ one
python3 -u src/main_incremental.py --gpu 1 --results-path exp1_4 --eval-on-train --batch-size 128 --wandb-project exp1_4_lwf --nepoch 200 --approach lwf --dkd-control adaptive --dkd-m 2.0 --dkd-switch ada_switch_4 --dkd-shape one --distill-percent 0.2 --ikr-control deterministic --ikr-m 2.0 --ikr-switch all --pk-approach average
# 20%KD ada_switch_5 _ one
python3 -u src/main_incremental.py --gpu 1 --results-path exp1_4 --eval-on-train --batch-size 128 --wandb-project exp1_4_lwf --nepoch 200 --approach lwf --dkd-control adaptive --dkd-m 2.0 --dkd-switch ada_switch_5 --dkd-shape one --distill-percent 0.2 --ikr-control deterministic --ikr-m 2.0 --ikr-switch all --pk-approach average