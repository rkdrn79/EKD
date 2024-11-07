## deterministic

# 20%KD
python3 -u src/main_incremental.py --gpu 1 --results-path exp1_4  --eval-on-train --batch-size 128 --wandb-project exp1_4_lwf --nepoch 200 --approach lwf --dkd-control deterministic --dkd-m 2.0 --dkd-switch cycle --dkd-shape one --distill-percent 0.2 --ikr-control deterministic --ikr-m 2.0 --ikr-switch all --pk-approach average
python3 -u src/main_incremental.py --gpu 1 --results-path exp1_4  --eval-on-train --batch-size 128 --wandb-project exp1_4_lwf --nepoch 200 --approach lwf --dkd-control deterministic --dkd-m 2.0 --dkd-switch first --dkd-shape one --distill-percent 0.2 --ikr-control deterministic --ikr-m 2.0 --ikr-switch all --pk-approach average
python3 -u src/main_incremental.py --gpu 1 --results-path exp1_4  --eval-on-train --batch-size 128 --wandb-project exp1_4_lwf --nepoch 200 --approach lwf --dkd-control deterministic --dkd-m 2.0 --dkd-switch mid --dkd-shape one --distill-percent 0.2 --ikr-control deterministic --ikr-m 2.0 --ikr-switch all --pk-approach average
# python3 -u src/main_incremental.py --gpu 0 --results-path exp1_4  --eval-on-train --batch-size 128 --wandb-project exp1_4_lwf --nepoch 200 --approach lwf --dkd-control deterministic --dkd-m 2.0 --dkd-switch end --dkd-shape one --distill-percent 0.2 --ikr-control deterministic --ikr-m 2.0 --ikr-switch all --pk-approach average
python3 -u src/main_incremental.py --gpu 2 --results-path exp1_4  --eval-on-train --batch-size 128 --wandb-project exp1_4_lwf --nepoch 200 --approach lwf --dkd-control deterministic --dkd-m 2.0 --dkd-switch first_end --dkd-shape one --distill-percent 0.2 --ikr-control deterministic --ikr-m 2.0 --ikr-switch all --pk-approach average
# python3 -u src/main_incremental.py --gpu 2 --results-path exp1_4  --eval-on-train --batch-size 128 --wandb-project exp1_4_lwf --nepoch 200 --approach lwf --dkd-control deterministic --dkd-m 2.0 --dkd-switch ten_to_ten --dkd-shape one --distill-percent 0.2 --ikr-control deterministic --ikr-m 2.0 --ikr-switch all --pk-approach average

## adaptive
# 100%KD ada_switch_1 _ one
python3 -u src/main_incremental.py --gpu 1 --results-path exp1_4 --eval-on-train --batch-size 128 --wandb-project exp1_3_lwf --nepoch 200 --approach lwf --dkd-control adaptive --dkd-m 2.0 --dkd-switch ada_switch_1 --dkd-shape one --distill-percent 1 --ikr-control deterministic --ikr-m 2.0 --ikr-switch all --pk-approach average
# 20%KD ada_switch_2 _ one
python3 -u src/main_incremental.py --gpu 1 --results-path exp1_4 --eval-on-train --batch-size 128 --wandb-project exp1_3_lwf --nepoch 200 --approach lwf --dkd-control adaptive --dkd-m 2.0 --dkd-switch ada_switch_2 --dkd-shape one --distill-percent 0.2 --ikr-control deterministic --ikr-m 2.0 --ikr-switch all --pk-approach average
# 20%KD ada_switch_3 _ one
python3 -u src/main_incremental.py --gpu 1 --results-path exp1_4 --eval-on-train --batch-size 128 --wandb-project exp1_3_lwf --nepoch 200 --approach lwf --dkd-control adaptive --dkd-m 2.0 --dkd-switch ada_switch_3 --dkd-shape one --distill-percent 0.2 --ikr-control deterministic --ikr-m 2.0 --ikr-switch all --pk-approach average
# 20%KD ada_switch_4 _ one
python3 -u src/main_incremental.py --gpu 1 --results-path exp1_4 --eval-on-train --batch-size 128 --wandb-project exp1_3_lwf --nepoch 200 --approach lwf --dkd-control adaptive --dkd-m 2.0 --dkd-switch ada_switch_4 --dkd-shape one --distill-percent 0.2 --ikr-control deterministic --ikr-m 2.0 --ikr-switch all --pk-approach average
# 20%KD ada_switch_5 _ one
python3 -u src/main_incremental.py --gpu 1 --results-path exp1_4 --eval-on-train --batch-size 128 --wandb-project exp1_3_lwf --nepoch 200 --approach lwf --dkd-control adaptive --dkd-m 2.0 --dkd-switch ada_switch_5 --dkd-shape one --distill-percent 0.2 --ikr-control deterministic --ikr-m 2.0 --ikr-switch all --pk-approach average