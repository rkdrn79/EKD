# lwf
# 100% KD
# python3 -u src/main_incremental.py --gpu 1 --dataset cifar100 --results-path vit --network Vit_tiny_4_augreg_32 --eval-on-train --batch-size 512 --wandb-project vit_test --nepoch 200 --approach lwf --dkd-control deterministic --dkd-m 1.0 --dkd-switch all --dkd-shape one --distill-percent 1.0 --ikr-control none;
# finetuning
# python3 -u src/main_incremental.py --gpu 0 --dataset cifar100 --results-path vit --network Vit_tiny_4_augreg_32 --eval-on-train --batch-size 512 --wandb-project vit_test --nepoch 200 --approach lwf --dkd-control deterministic --dkd-m 1.0 --dkd-switch none --dkd-shape one --distill-percent 1.0 --ikr-control none;
# # 1-1
# python3 -u src/main_incremental.py --gpu 1 --dataset cifar100 --results-path vit --network Vit_tiny_4_augreg_32 --eval-on-train --batch-size 512 --wandb-project vit_test --nepoch 200 --approach lwf --dkd-control deterministic --dkd-m 2.0 --dkd-switch first_end --dkd-shape one --distill-percent 0.2 --ikr-control none;
# # 1-3
# python3 -u src/main_incremental.py --gpu 0 --dataset cifar100 --results-path vit --network Vit_tiny_4_augreg_32 --eval-on-train --batch-size 512 --wandb-project vit_test --nepoch 200 --approach lwf --dkd-control adaptive --dkd-m 2.0 --dkd-switch ada_switch_3 --dkd-shape one --distill-percent 0.2 --ikr-control none;

# # icarl
# 100% KD
# python3 -u src/main_incremental.py --gpu 0 --dataset cifar100 --results-path vit --network Vit_tiny_4_augreg_32 --eval-on-train --batch-size 512 --wandb-project vit_test --nepoch 200 --approach icarl --dkd-control deterministic --dkd-m 1.0 --dkd-switch all --dkd-shape one --distill-percent 1.0 --ikr-control none;
# finetuning
# python3 -u src/main_incremental.py --gpu 1 --dataset cifar100 --results-path vit --network Vit_tiny_4_augreg_32 --eval-on-train --batch-size 512 --wandb-project vit_test --nepoch 200 --approach icarl --dkd-control deterministic --dkd-m 1.0 --dkd-switch none --dkd-shape one --distill-percent 1.0 --ikr-control none;
# 1-1
python3 -u src/main_incremental.py --gpu 0 --dataset cifar100 --results-path vit --network Vit_tiny_4_augreg_32 --eval-on-train --batch-size 512 --wandb-project vit_test --nepoch 200 --approach icarl --dkd-control deterministic --dkd-m 0.5 --dkd-switch cycle --dkd-shape one --distill-percent 0.2 --ikr-control none
# # 1-3
python3 -u src/main_incremental.py --gpu 1 --dataset cifar100 --results-path vit --network Vit_tiny_4_augreg_32 --eval-on-train --batch-size 512 --wandb-project vit_test --nepoch 200 --approach icarl --dkd-control adaptive --dkd-m 0.5 --dkd-switch ada_switch_2 --dkd-shape one --distill-percent 0.2 --ikr-control none

# lucir
# 100% KD
python3 -u src/main_incremental.py --gpu 1 --dataset cifar100 --results-path vit --network Vit_tiny_4_augreg_32 --eval-on-train --batch-size 512 --wandb-project vit_test --nepoch 200 --approach lucir --dkd-control deterministic --dkd-m 1.0 --dkd-switch all --dkd-shape one --distill-percent 1.0 --ikr-control none;
# finetuning
python3 -u src/main_incremental.py --gpu 0 --dataset cifar100 --results-path vit --network Vit_tiny_4_augreg_32 --eval-on-train --batch-size 512 --wandb-project vit_test --nepoch 200 --approach lucir --dkd-control deterministic --dkd-m 1.0 --dkd-switch none --dkd-shape one --distill-percent 1.0 --ikr-control none;
# 1-1
python3 -u src/main_incremental.py --gpu 1 --dataset cifar100 --results-path vit --network Vit_tiny_4_augreg_32 --eval-on-train --batch-size 512 --wandb-project vit_test --nepoch 200 --approach lucir --dkd-control deterministic --dkd-m 0.5 --dkd-switch ten_to_ten --dkd-shape one --distill-percent 0.2 --ikr-control none;
# 1-3
python3 -u src/main_incremental.py --gpu 0 --dataset cifar100 --results-path vit --network Vit_tiny_4_augreg_32 --eval-on-train --batch-size 512 --wandb-project vit_test --nepoch 200 --approach lucir --dkd-control adaptive --dkd-m 0.5 --dkd-switch ada_switch_2 --dkd-shape one --distill-percent 0.2 --ikr-control none;

# lwm
# 100% KD
python3 -u src/main_incremental.py --gpu 1 --dataset cifar100 --results-path vit --network Vit_tiny_4_augreg_32 --eval-on-train --batch-size 512 --wandb-project vit_test --nepoch 200 --approach lwm --dkd-control deterministic --dkd-m 1.0 --dkd-switch all --dkd-shape one --distill-percent 1.0 --ikr-control none;
# finetuning
python3 -u src/main_incremental.py --gpu 1 --dataset cifar100 --results-path vit --network Vit_tiny_4_augreg_32 --eval-on-train --batch-size 512 --wandb-project vit_test --nepoch 200 --approach lwm --dkd-control deterministic --dkd-m 1.0 --dkd-switch none --dkd-shape one --distill-percent 1.0 --ikr-control none;
# 1-1
python3 -u src/main_incremental.py --gpu 0 --dataset cifar100 --results-path vit --network Vit_tiny_4_augreg_32 --eval-on-train --batch-size 512 --wandb-project vit_test --nepoch 200 --approach lwm --dkd-control deterministic --dkd-m 2.0 --dkd-switch ten_to_ten --dkd-shape one --distill-percent 0.2 --ikr-control none
# 1-3
python3 -u src/main_incremental.py --gpu 0 --dataset cifar100 --results-path vit --network Vit_tiny_4_augreg_32 --eval-on-train --batch-size 512 --wandb-project vit_test --nepoch 200 --approach lwm --dkd-control adaptive --dkd-m 2.0 --dkd-switch ada_switch_3 --dkd-shape one --distill-percent 0.2 --ikr-control none
