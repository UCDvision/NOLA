dset='CIFAR10'
arch='large'
train_type='nola_mlp'
kshot=5
num_cls=10
lr=5e-3
ka=1024
rank=1
# Loop - 4 diff seeds for dataset kshot sampling (idx j) --> 3 runs (with diff seeds for network init, idx k)
for k in {0..0}
do
  for j in {0..0}
  do
    gpu=$(($j+0))
    CUDA_VISIBLE_DEVICES="$gpu" python train_timm.py \
      --train_type "$train_type" \
      --num_classes "$num_cls" \
      --rank $rank \
      --ka $ka \
      --kb $ka \
      --lr $lr \
      --vit $arch \
      --save-weights \
      -bs 16 \
      --kshot "$kshot" \
      --kshot_seed "$j" \
      --epochs 50 --warmup-epochs 0 \
      --train_data_path /home/navaneet/data/"$dset" \
      --val_data_path /home/navaneet/data/"$dset" \
      --outdir ./exp/e001_"$arch"_ft_"$train_type"_"$dset"_r"$rank"_k"$ka"_lr"$lr"/"$kshot"shot/v"$j"/run"$k"
#      --outdir ./exp/temp
#      --eval \
  done
  wait
done
