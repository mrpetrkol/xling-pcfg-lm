#!/bin/bash

trap 'echo "Stopping..."; kill 0' SIGINT

run_experiment() {
    local gpu=$1
    local experiment=$2
    local rule=$3
    local prefix=$4
    shift 4
    local ps=("$@")

    for p in "${ps[@]}"; do
        echo "Running: experiment $experiment with rule=$rule and p=$p on GPU $gpu"
        CUDA_VISIBLE_DEVICES=$gpu python -m lm_pcfg --experiment "$experiment" --p "$p" --rule "$rule" 2>&1 | sed "s/^/$prefix /"
    done
}

p_values=(0.1 0.2 0.5 0.8 0.9)

run_experiment 0 1 "S -> ATS DOT" "process 0:" "${p_values[@]}" &
run_experiment 1 1 "S -> NP VP" "process 1:" "${p_values[@]}" &

wait