#!/bin/bash
#SBATCH --job-name=orchestrator_golden
#SBATCH --time=00:10:00
#SBATCH --mem=1G
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --partition=compute
#SBATCH --account=proj123
#SBATCH --output=/golden/GOLDEN/logs/orchestrator_%j.out

module load python/3.11

export MY_VAR="value"

STEP_COUNT=$1
LOOP_COUNT=$2
INNER_COUNT=${3:-0}
SELF=$(realpath "$0")
TOTAL_STEPS=4
JOB_DIR="/golden/GOLDEN"

while [ "$STEP_COUNT" -lt "$TOTAL_STEPS" ]; do

  if [ "$STEP_COUNT" -eq 0 ]; then
    # Step: make
    RESULT=$(sbatch "SCRIPT_DIR/make.sh")
    JOB_ID=$(echo $RESULT | awk '{print $NF}')
    echo "Submitted make: job $JOB_ID"
    STEP_COUNT=1
    if [ -n "$JOB_ID" ]; then
      sbatch --dependency=afterok:$JOB_ID "$SELF" $STEP_COUNT $LOOP_COUNT
    else
      sbatch "$SELF" $STEP_COUNT $LOOP_COUNT
    fi
    exit 0

  elif [ "$STEP_COUNT" -eq 1 ]; then
    # Step: evaluate
    N_OPEN=$(python -m f3dasm.pipeline.count_open --job-dir="$JOB_DIR" --project-dir="eval")
    if [ "$N_OPEN" -gt 0 ]; then
      ARRAY_MAX=$(( (N_OPEN < 100 ? N_OPEN : 100) - 1 ))
      RESULT=$(sbatch --array=0-${ARRAY_MAX}%32 "SCRIPT_DIR/evaluate.sh")
      JOB_ID=$(echo $RESULT | awk '{print $NF}')
      echo "Submitted evaluate: job $JOB_ID (array 0-$ARRAY_MAX)"
    else
      echo "Skipping evaluate: no open experiments"
      JOB_ID=""
    fi
    STEP_COUNT=2
    if [ -n "$JOB_ID" ]; then
      sbatch --dependency=afterok:$JOB_ID "$SELF" $STEP_COUNT $LOOP_COUNT
    else
      sbatch "$SELF" $STEP_COUNT $LOOP_COUNT
    fi
    exit 0

  elif [ "$STEP_COUNT" -eq 2 ]; then
    # Loop: 3 iterations
    if [ "$LOOP_COUNT" -lt 3 ]; then
      export F3DASM_ITERATION=$LOOP_COUNT
      if [ "$INNER_COUNT" -eq 0 ]; then
        # Inner step: sample
        RESULT=$(sbatch --export=ALL "SCRIPT_DIR/loop2_sample.sh")
        JOB_ID=$(echo $RESULT | awk '{print $NF}')
        echo "  Submitted sample (iter $LOOP_COUNT): job $JOB_ID"
        if [ -n "$JOB_ID" ]; then
          sbatch --dependency=afterany:$JOB_ID "$SELF" $STEP_COUNT $LOOP_COUNT 1
        else
          sbatch "$SELF" $STEP_COUNT $LOOP_COUNT 1
        fi
        exit 0
      elif [ "$INNER_COUNT" -eq 1 ]; then
        # Inner step: run
        N_OPEN=$(python -m f3dasm.pipeline.count_open --job-dir="$JOB_DIR" --project-dir=".")
        if [ "$N_OPEN" -gt 0 ]; then
          ARRAY_MAX=$(( (N_OPEN < 100 ? N_OPEN : 100) - 1 ))
          RESULT=$(sbatch --array=0-${ARRAY_MAX}%32 --export=ALL "SCRIPT_DIR/loop2_run.sh")
          JOB_ID=$(echo $RESULT | awk '{print $NF}')
          echo "Submitted   run (iter $LOOP_COUNT): job $JOB_ID (array 0-$ARRAY_MAX)"
        else
          echo "Skipping   run (iter $LOOP_COUNT): no open experiments"
          JOB_ID=""
        fi
        if [ -n "$JOB_ID" ]; then
          sbatch --dependency=afterok:$JOB_ID "$SELF" $STEP_COUNT $((LOOP_COUNT + 1)) 0
        else
          sbatch "$SELF" $STEP_COUNT $((LOOP_COUNT + 1)) 0
        fi
        exit 0
      fi
    else
      # Loop done — advance to next element
      LOOP_COUNT=0
      INNER_COUNT=0
      STEP_COUNT=3
      continue
    fi

  elif [ "$STEP_COUNT" -eq 3 ]; then
    # Step: collect
    RESULT=$(sbatch "SCRIPT_DIR/collect.sh")
    JOB_ID=$(echo $RESULT | awk '{print $NF}')
    echo "Submitted collect: job $JOB_ID"
    STEP_COUNT=4
    if [ -n "$JOB_ID" ]; then
      sbatch --dependency=afterok:$JOB_ID "$SELF" $STEP_COUNT $LOOP_COUNT
    else
      sbatch "$SELF" $STEP_COUNT $LOOP_COUNT
    fi
    exit 0

  fi
done

echo "Pipeline complete."
