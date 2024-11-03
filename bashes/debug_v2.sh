# # lwf 테스트
# ## --results-path exp1_2

# ## 20%KD cycle _ linear_increase
# python3 -u src/main_incremental.py --gpu 0 --results-path exp1_2 --eval-on-train --batch-size 128 --wandb-project exp1_2_lwf --nepoch 200 --approach lwf --dkd-control deterministic --dkd-m 2.0 --dkd-switch cycle --dkd-shape linear_increase --distill-percent 0.2 --ikr-control none
# ## 20%KD first _ linear_decrease
# python3 -u src/main_incremental.py --gpu 0 --results-path exp1_2 --eval-on-train --batch-size 128 --wandb-project exp1_2_lwf --nepoch 200 --approach lwf --dkd-control deterministic --dkd-m 2.0 --dkd-switch first --dkd-shape linear_decrease --distill-percent 0.2 --ikr-control none
# ## 20%KD mid _ concaved_increase
# python3 -u src/main_incremental.py --gpu 0 --results-path exp1_2 --eval-on-train --batch-size 128 --wandb-project exp1_2_lwf --nepoch 200 --approach lwf --dkd-control deterministic --dkd-m 2.0 --dkd-switch mid --dkd-shape concaved_increase --distill-percent 0.2 --ikr-control none
# ## 20%KD end _ concaved_decrease
# python3 -u src/main_incremental.py --gpu 0 --results-path exp1_2 --eval-on-train --batch-size 128 --wandb-project exp1_2_lwf --nepoch 200 --approach lwf --dkd-control deterministic --dkd-m 2.0 --dkd-switch end --dkd-shape concaved_decrease --distill-percent 0.2 --ikr-control none
# ## 20%KD first_end _ convexed_increase
# python3 -u src/main_incremental.py --gpu 0 --results-path exp1_2 --eval-on-train --batch-size 128 --wandb-project exp1_2_lwf --nepoch 200 --approach lwf --dkd-control deterministic --dkd-m 2.0 --dkd-switch first_end --dkd-shape convexed_increase --distill-percent 0.2 --ikr-control none
# ## 20%KD ten_to_ten _ convexed_decrease
# python3 -u src/main_incremental.py --gpu 0 --results-path exp1_2 --eval-on-train --batch-size 128 --wandb-project exp1_2_lwf --nepoch 200 --approach lwf --dkd-control deterministic --dkd-m 2.0 --dkd-switch ten_to_ten --dkd-shape convexed_decrease --distill-percent 0.2 --ikr-control none


# dkd_m = 1.0
## 100%KD
python3 -u src/main_incremental.py --gpu 0 --results-path exp1_1  --eval-on-train --batch-size 128 --wandb-project exp1_1_lwf --nepoch 200 --approach lwf --dkd-control deterministic --dkd-m 1.0 --dkd-switch all --dkd-shape one --distill-percent 1 --ikr-control none
## 80%KD
python3 -u src/main_incremental.py --gpu 0 --results-path exp1_1  --eval-on-train --batch-size 128 --wandb-project exp1_1_lwf --nepoch 200 --approach lwf --dkd-control deterministic --dkd-m 1.0 --dkd-switch cycle --dkd-shape one --distill-percent 0.8 --ikr-control none
python3 -u src/main_incremental.py --gpu 0 --results-path exp1_1  --eval-on-train --batch-size 128 --wandb-project exp1_1_lwf --nepoch 200 --approach lwf --dkd-control deterministic --dkd-m 1.0 --dkd-switch first --dkd-shape one --distill-percent 0.8 --ikr-control none
python3 -u src/main_incremental.py --gpu 0 --results-path exp1_1  --eval-on-train --batch-size 128 --wandb-project exp1_1_lwf --nepoch 200 --approach lwf --dkd-control deterministic --dkd-m 1.0 --dkd-switch mid --dkd-shape one --distill-percent 0.8 --ikr-control none
python3 -u src/main_incremental.py --gpu 0 --results-path exp1_1  --eval-on-train --batch-size 128 --wandb-project exp1_1_lwf --nepoch 200 --approach lwf --dkd-control deterministic --dkd-m 1.0 --dkd-switch end --dkd-shape one --distill-percent 0.8 --ikr-control none
python3 -u src/main_incremental.py --gpu 0 --results-path exp1_1  --eval-on-train --batch-size 128 --wandb-project exp1_1_lwf --nepoch 200 --approach lwf --dkd-control deterministic --dkd-m 1.0 --dkd-switch first_end --dkd-shape one --distill-percent 0.8 --ikr-control none
python3 -u src/main_incremental.py --gpu 0 --results-path exp1_1  --eval-on-train --batch-size 128 --wandb-project exp1_1_lwf --nepoch 200 --approach lwf --dkd-control deterministic --dkd-m 1.0 --dkd-switch ten_to_ten --dkd-shape one --distill-percent 0.8 --ikr-control none
## 60%KD
python3 -u src/main_incremental.py --gpu 0 --results-path exp1_1  --eval-on-train --batch-size 128 --wandb-project exp1_1_lwf --nepoch 200 --approach lwf --dkd-control deterministic --dkd-m 1.0 --dkd-switch cycle --dkd-shape one --distill-percent 0.6 --ikr-control none
python3 -u src/main_incremental.py --gpu 0 --results-path exp1_1  --eval-on-train --batch-size 128 --wandb-project exp1_1_lwf --nepoch 200 --approach lwf --dkd-control deterministic --dkd-m 1.0 --dkd-switch first --dkd-shape one --distill-percent 0.6 --ikr-control none
python3 -u src/main_incremental.py --gpu 0 --results-path exp1_1  --eval-on-train --batch-size 128 --wandb-project exp1_1_lwf --nepoch 200 --approach lwf --dkd-control deterministic --dkd-m 1.0 --dkd-switch mid --dkd-shape one --distill-percent 0.6 --ikr-control none
python3 -u src/main_incremental.py --gpu 0 --results-path exp1_1  --eval-on-train --batch-size 128 --wandb-project exp1_1_lwf --nepoch 200 --approach lwf --dkd-control deterministic --dkd-m 1.0 --dkd-switch end --dkd-shape one --distill-percent 0.6 --ikr-control none
python3 -u src/main_incremental.py --gpu 0 --results-path exp1_1  --eval-on-train --batch-size 128 --wandb-project exp1_1_lwf --nepoch 200 --approach lwf --dkd-control deterministic --dkd-m 1.0 --dkd-switch first_end --dkd-shape one --distill-percent 0.6 --ikr-control none
python3 -u src/main_incremental.py --gpu 0 --results-path exp1_1  --eval-on-train --batch-size 128 --wandb-project exp1_1_lwf --nepoch 200 --approach lwf --dkd-control deterministic --dkd-m 1.0 --dkd-switch ten_to_ten --dkd-shape one --distill-percent 0.6 --ikr-control none
## 40%KD
python3 -u src/main_incremental.py --gpu 0 --results-path exp1_1  --eval-on-train --batch-size 128 --wandb-project exp1_1_lwf --nepoch 200 --approach lwf --dkd-control deterministic --dkd-m 1.0 --dkd-switch cycle --dkd-shape one --distill-percent 0.4 --ikr-control none
python3 -u src/main_incremental.py --gpu 0 --results-path exp1_1  --eval-on-train --batch-size 128 --wandb-project exp1_1_lwf --nepoch 200 --approach lwf --dkd-control deterministic --dkd-m 1.0 --dkd-switch first --dkd-shape one --distill-percent 0.4 --ikr-control none
python3 -u src/main_incremental.py --gpu 0 --results-path exp1_1  --eval-on-train --batch-size 128 --wandb-project exp1_1_lwf --nepoch 200 --approach lwf --dkd-control deterministic --dkd-m 1.0 --dkd-switch mid --dkd-shape one --distill-percent 0.4 --ikr-control none
python3 -u src/main_incremental.py --gpu 0 --results-path exp1_1  --eval-on-train --batch-size 128 --wandb-project exp1_1_lwf --nepoch 200 --approach lwf --dkd-control deterministic --dkd-m 1.0 --dkd-switch end --dkd-shape one --distill-percent 0.4 --ikr-control none
python3 -u src/main_incremental.py --gpu 0 --results-path exp1_1  --eval-on-train --batch-size 128 --wandb-project exp1_1_lwf --nepoch 200 --approach lwf --dkd-control deterministic --dkd-m 1.0 --dkd-switch first_end --dkd-shape one --distill-percent 0.4 --ikr-control none
python3 -u src/main_incremental.py --gpu 0 --results-path exp1_1  --eval-on-train --batch-size 128 --wandb-project exp1_1_lwf --nepoch 200 --approach lwf --dkd-control deterministic --dkd-m 1.0 --dkd-switch ten_to_ten --dkd-shape one --distill-percent 0.4 --ikr-control none
## 20%KD
python3 -u src/main_incremental.py --gpu 0 --results-path exp1_1  --eval-on-train --batch-size 128 --wandb-project exp1_1_lwf --nepoch 200 --approach lwf --dkd-control deterministic --dkd-m 1.0 --dkd-switch cycle --dkd-shape one --distill-percent 0.2 --ikr-control none
python3 -u src/main_incremental.py --gpu 0 --results-path exp1_1  --eval-on-train --batch-size 128 --wandb-project exp1_1_lwf --nepoch 200 --approach lwf --dkd-control deterministic --dkd-m 1.0 --dkd-switch first --dkd-shape one --distill-percent 0.2 --ikr-control none
python3 -u src/main_incremental.py --gpu 0 --results-path exp1_1  --eval-on-train --batch-size 128 --wandb-project exp1_1_lwf --nepoch 200 --approach lwf --dkd-control deterministic --dkd-m 1.0 --dkd-switch mid --dkd-shape one --distill-percent 0.2 --ikr-control none
python3 -u src/main_incremental.py --gpu 0 --results-path exp1_1  --eval-on-train --batch-size 128 --wandb-project exp1_1_lwf --nepoch 200 --approach lwf --dkd-control deterministic --dkd-m 1.0 --dkd-switch end --dkd-shape one --distill-percent 0.2 --ikr-control none
python3 -u src/main_incremental.py --gpu 0 --results-path exp1_1  --eval-on-train --batch-size 128 --wandb-project exp1_1_lwf --nepoch 200 --approach lwf --dkd-control deterministic --dkd-m 1.0 --dkd-switch first_end --dkd-shape one --distill-percent 0.2 --ikr-control none
python3 -u src/main_incremental.py --gpu 0 --results-path exp1_1  --eval-on-train --batch-size 128 --wandb-project exp1_1_lwf --nepoch 200 --approach lwf --dkd-control deterministic --dkd-m 1.0 --dkd-switch ten_to_ten --dkd-shape one --distill-percent 0.2 --ikr-control none
