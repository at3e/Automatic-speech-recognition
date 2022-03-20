#!/usr/bin/bash

on_error(){
  >&2 echo "Error: $1"
  exit 2
}

brewpath=(/idiap/group/speech/local/bin/brew shellenv)
root="/idiap/temp/asaha/wav2vec/"
get_free_port="${root}/scripts/get_free_port.py"

[ -e "${get_free_port}" ] || on_error "${get_free_port} does not exist!"


################### Training Parameters ####################
prefix=w2v
pt_model=wav2vec_small
dataset=main
n_nodes=4
gpu_queue=sgpu
max_tokens=800000
n_updates=25000
update_freq=8
tdnnf_grad_mult=20.0
mask_updates=23000
freeze_updates=500
conf

. parse_options.sh || exit 1;


echo $n_nodes
echo $update_freq
echo $n_updates
manifest="${root}/data/${lang}/${dataset}"

model="${prefix}_${lang}_${pt_model}_ftctc_${dataset}_u$((n_updates / 1000))k_b${n_nodes}x>
logroot="${root}/models/${model}/logs"
mkdir -p "logs/train"
mkdir -p "logs/eval"
mkdir -p "${logroot}"

echo $manifest $lang $conf $model
echo $gpu_queue
echo $model
#################### Training Parameters (end) ##############

#################### Prepare Commands ####################

info_file=$(mktemp -p "${root}/jobs/master_info") &&
>&2 echo "Submit distributed training jobs" &&
>&2 echo "Master node info file: ${info_file}" &&
>&2 echo "Command:" &&
>&2 echo "  ${cmd}" &&
for (( node_id=0; node_id<${n_nodes}; node_id++))
do
  echo ${node_id}
  qsub \
    -N w2v_ft_base \
    -S /bin/bash -cwd -V \
    -P shissm \
    -l pytorch,h=vgne*,gpumem=11,${gpu_queue} \
    -o 'logs/train/$JOB_NAME_$JOB_ID.stdout' \
    -e 'logs/train/$JOB_NAME_$JOB_ID.stderr' \
    -M avyas@idiap.ch -m ea \
    <<EOF
source ${HOME}/.bashrc
>&2 echo "node ($node_id) @ \$(hostname)" &&
if [ "${node_id}" -eq 0 ]
then
  master_host=\$(hostname) &&
  master_port=\$(${get_free_port})
  echo "\$master_host \$master_port" > ${info_file}
else
  while [ "\$(wc -l ${info_file} | awk '{print \$1}')" -lt 1 ]
  do
    >&2 echo "wait for master info..."
    sleep 1
  done
  read master_host master_port < ${info_file}
fi
>&2 echo "master at \${master_host}:\${master_port}" &&
>&2 echo "${cmd}" &&
time python -m torch.distributed.launch --use_env --nproc_per_node=1 \
  --nnodes=${n_nodes} --node_rank=${node_id} \
  --master_addr="\${master_host}" --master_port="\${master_port}" \
  ${cmd} \
  | tee -a "${logroot}/log_${node_id}"
EOF
done
