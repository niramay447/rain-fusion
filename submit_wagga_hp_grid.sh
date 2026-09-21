#!/bin/bash
# Fan out the Wagga gauge-only HP-tune grid on Gadi.
# Fixed: log_transform=True, weighted_loss_alpha=0.5, num_layers=5.
# Grid: lr {1e-3, 3e-4, 1e-4} x hidden {32, 64} x dropout {0, 0.15} = 12 cells.
# Cell (lr=0.0003,h=32,d=0.15) reproduces wagga_gauge_log_inck_0.5 (pearson 0.658) = anchor.
# Run from PROJECT_DIR on Gadi:  bash submit_wagga_hp_grid.sh
set -euo pipefail

LRS=(0.001 0.0003 0.0001)
HIDS=(32 64)
DROPS=(0 0.15)

for LR in "${LRS[@]}"; do
  for HID in "${HIDS[@]}"; do
    for DROP in "${DROPS[@]}"; do
      SLUG="lr${LR}_h${HID}_d${DROP}"
      qsub -N "wgh_${SLUG}" \
           -v "LR=${LR},HID=${HID},DROP=${DROP}" \
           -o "logs/wagga_gauge_hp_${SLUG}.log" \
           run_wagga_gauge_hp.pbs
      echo "submitted ${SLUG}"
    done
  done
done
echo "12 cells submitted."
