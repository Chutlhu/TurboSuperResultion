source /home/dicarlo_d/Documents/Code/TurboSuperResultion/venv/bin/activate
data_dir=/home/dicarlo_d/Documents/Code/TurboSuperResultion/.cache/Turb2D.hdf5

log_dir=logs_comparison
res_dir=results_comparison

mkdir -p figures
mkdir -p $res_dir

max_epoch=7000
test_ds=1
mlp_layers_dim=256
do_rff=true

lam_sdiv=0
lam_grads=0
lam_curl=0
lam_sfn=0
ni=4


for run in 333
do

for train_ds in 4
do 


for do_dvf in false
do

val_ds=$train_ds

for mlp_layers_num in 10
do

for rff_num in 512
do

for lam_sdiv in 1e-3
do

for lam_grads in 1e-5
do

for lam_curl in 1e-8
do

for lam_sfn in 1e-3
do

name="RFFMLP_run:$run-ds:$train_ds-mlp:$mlp_layers_num-rff_num:$rff_num-df:$do_dvf-lsdv:$lam_sdiv-lgrd:$lam_grads-lcrl:$lam_curl-lsfn:$lam_sfn-ni:$ni"
echo $name

python main.py  --exp_name $name                        \
                --logs_dir $log_dir                     \
                --res_dir $res_dir                  \
                --dataset Turb2D --data_dir $data_dir   \
                --time_idx $run                         \
                --train_downsampling $train_ds          \
                --val_downsampling   $val_ds            \
                --test_downsampling  $test_ds           \
                --num_workers 1                         \
                --name  $name                           \
                --mlp_layers_num $mlp_layers_num        \
                --mlp_layers_dim $mlp_layers_dim        \
                --do_rff $do_rff --rff_num $rff_num     \
                --do_divfree $do_dvf                    \
                --lam_sdiv $lam_sdiv                    \
                --lam_sfn $lam_sfn                      \
                --lam_grads $lam_grads                  \
                --lam_curl $lam_curl                    \
                --lam_pde 0                             \
                --lam_weight 1e-4                       \
                --sfn_num_centers 666                   \
                --sfn_num_increments $ni                \
                --sfn_patch_dim   666                   \
                --gpus 1 --max_epochs $max_epoch        \
                --log_every_n_steps 10 --check_val_every_n_epoch 20

done
done
done
done
done
done
done
done
done