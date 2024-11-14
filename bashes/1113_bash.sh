# imagenet icarl

python3 -u src/main_incremental.py --gpu 0 --dataset imagenet_32_reduced --results-path imagenet_exp1_1 --network resnet32 --eval-on-train --batch-size 512 --wandb-project imagenet32_test --nepoch 200 --approach icarl --dkd-control deterministic --dkd-m 0.5 --dkd-switch cycle --dkd-shape one --distill-percent 0.2 --ikr-control none


python3 -u src/main_incremental.py --gpu 1 --dataset cifar100 --results-path vit --network Vit_tiny_4_augreg_32 --eval-on-train --batch-size 512 --wandb-project vit_test --nepoch 200 --approach lwf --dkd-control deterministic --dkd-m 2.0 --dkd-switch ten_to_ten --dkd-shape one --distill-percent 0.2 --ikr-control none;
# python3 -u src/main_incremental.py --gpu 1 --dataset cifar100 --results-path vit --network Vit_tiny_4_augreg_32 --eval-on-train --batch-size 512 --wandb-project vit_test --nepoch 200 --approach icarl --dkd-control deterministic --dkd-m 2.0 --dkd-switch ten_to_ten --dkd-shape one --distill-percent 0.2 --ikr-control none;

python3 -u src/main_incremental.py --gpu 0 --dataset cifar100 --results-path vit --network Vit_tiny_4_augreg_32 --eval-on-train --batch-size 512 --wandb-project vit_test --nepoch 200 --approach icarl --dkd-control adaptive --dkd-m 0.5 --dkd-switch ada_switch_3 --dkd-shape one --distill-percent 0.2 --ikr-control none;
python3 -u src/main_incremental.py --gpu 0 --dataset cifar100 --results-path vit --network Vit_tiny_4_augreg_32 --eval-on-train --batch-size 512 --wandb-project vit_test --nepoch 200 --approach lucir --dkd-control adaptive --dkd-m 0.5 --dkd-switch ada_switch_3 --dkd-shape one --distill-percent 0.2 --ikr-control none;
