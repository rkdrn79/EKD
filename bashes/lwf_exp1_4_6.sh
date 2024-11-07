## deterministic

# 20%KD
python3 -u src/main_incremental.py --gpu 2 --results-path exp1_4  --eval-on-train --batch-size 128 --wandb-project exp1_4_lwf --nepoch 200 --approach lwf --dkd-control deterministic --dkd-m 2.0 --dkd-switch cycle --dkd-shape one --distill-percent 0.2 --ikr-control deterministic --ikr-m 0.5 --ikr-switch all --pk-approach average
python3 -u src/main_incremental.py --gpu 2 --results-path exp1_4  --eval-on-train --batch-size 128 --wandb-project exp1_4_lwf --nepoch 200 --approach lwf --dkd-control deterministic --dkd-m 2.0 --dkd-switch first --dkd-shape one --distill-percent 0.2 --ikr-control deterministic --ikr-m 0.5 --ikr-switch all --pk-approach average
python3 -u src/main_incremental.py --gpu 2 --results-path exp1_4  --eval-on-train --batch-size 128 --wandb-project exp1_4_lwf --nepoch 200 --approach lwf --dkd-control deterministic --dkd-m 2.0 --dkd-switch mid --dkd-shape one --distill-percent 0.2 --ikr-control deterministic --ikr-m 0.5 --ikr-switch all --pk-approach average
python3 -u src/main_incremental.py --gpu 2 --results-path exp1_4  --eval-on-train --batch-size 128 --wandb-project exp1_4_lwf --nepoch 200 --approach lwf --dkd-control deterministic --dkd-m 2.0 --dkd-switch first_end --dkd-shape one --distill-percent 0.2 --ikr-control deterministic --ikr-m 0.5 --ikr-switch all --pk-approach average
python3 -u src/main_incremental.py --gpu 2 --results-path exp1_4  --eval-on-train --batch-size 128 --wandb-project exp1_4_lwf --nepoch 200 --approach lwf --dkd-control deterministic --dkd-m 2.0 --dkd-switch ten_to_ten --dkd-shape one --distill-percent 0.2 --ikr-control deterministic --ikr-m 0.5 --ikr-switch all --pk-approach average
