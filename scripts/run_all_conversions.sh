#!/bin/bash
set -e

DATA=/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/miaochenjin/WHA-MS/datafiles
OUT=$DATA/training/mmap-starting-fixed
LOG=$OUT/conversion.log

echo "========================================" | tee -a $LOG
echo "Starting all conversions at $(date)" | tee -a $LOG
echo "========================================" | tee -a $LOG

# WHAMSv2/BDT datasets (all events)
echo "=== WHAMSv2/BDT nugen_21217 ===" | tee -a $LOG
python scripts/convert_to_mmap.py --input $DATA/WHAMSv2/BDT/21217 --output $OUT/nugen_21217 2>&1 | tee -a $LOG
echo "Completed nugen_21217 at $(date)" | tee -a $LOG

echo "=== WHAMSv2/BDT nugen_21218 ===" | tee -a $LOG
python scripts/convert_to_mmap.py --input $DATA/WHAMSv2/BDT/21218 --output $OUT/nugen_21218 2>&1 | tee -a $LOG
echo "Completed nugen_21218 at $(date)" | tee -a $LOG

echo "=== WHAMSv2/BDT nugen_21219 ===" | tee -a $LOG
python scripts/convert_to_mmap.py --input $DATA/WHAMSv2/BDT/21219 --output $OUT/nugen_21219 2>&1 | tee -a $LOG
echo "Completed nugen_21219 at $(date)" | tee -a $LOG

echo "=== WHAMSv2/BDT corsika_22803 ===" | tee -a $LOG
python scripts/convert_to_mmap.py --input $DATA/WHAMSv2/BDT/22803 --output $OUT/corsika_22803 2>&1 | tee -a $LOG
echo "Completed corsika_22803 at $(date)" | tee -a $LOG

echo "=== WHAMSv2/BDT muongun_21316_extended ===" | tee -a $LOG
python scripts/convert_to_mmap.py --input $DATA/WHAMSv2/BDT/21316_extended --output $OUT/muongun_21316_extended 2>&1 | tee -a $LOG
echo "Completed muongun_21316_extended at $(date)" | tee -a $LOG

# WHAMSv1/BDT datasets (all events)
echo "=== WHAMSv1/BDT muongun_21315 ===" | tee -a $LOG
python scripts/convert_to_mmap.py --input $DATA/WHAMSv1/BDT/21315 --output $OUT/muongun_21315 2>&1 | tee -a $LOG
echo "Completed muongun_21315 at $(date)" | tee -a $LOG

echo "=== WHAMSv1/BDT muongun_21317 ===" | tee -a $LOG
python scripts/convert_to_mmap.py --input $DATA/WHAMSv1/BDT/21317 --output $OUT/muongun_21317 2>&1 | tee -a $LOG
echo "Completed muongun_21317 at $(date)" | tee -a $LOG

echo "=== WHAMSv1/BDT muongun_21318 ===" | tee -a $LOG
python scripts/convert_to_mmap.py --input $DATA/WHAMSv1/BDT/21318 --output $OUT/muongun_21318 2>&1 | tee -a $LOG
echo "Completed muongun_21318 at $(date)" | tee -a $LOG

# Starting-only datasets
echo "=== WHAMSv2/BDT nugen_22663_starting (--starting-only) ===" | tee -a $LOG
python scripts/convert_to_mmap.py --input $DATA/WHAMSv2/BDT/22663 --output $OUT/nugen_22663_starting --starting-only 2>&1 | tee -a $LOG
echo "Completed nugen_22663_starting at $(date)" | tee -a $LOG

echo "=== SIREN_inclusive_nue siren_inclusive_starting (--starting-only) ===" | tee -a $LOG
python scripts/convert_to_mmap.py --input $DATA/SIREN_inclusive_nue/BDT --output $OUT/siren_inclusive_starting --starting-only 2>&1 | tee -a $LOG
echo "Completed siren_inclusive_starting at $(date)" | tee -a $LOG

echo "=== SIREN_charms_nue siren_charms_starting (--starting-only) ===" | tee -a $LOG
python scripts/convert_to_mmap.py --input $DATA/SIREN_charms_nue --output $OUT/siren_charms_starting --starting-only 2>&1 | tee -a $LOG
echo "Completed siren_charms_starting at $(date)" | tee -a $LOG

echo "========================================" | tee -a $LOG
echo "All conversions completed at $(date)" | tee -a $LOG
echo "========================================" | tee -a $LOG

# List results
echo "" | tee -a $LOG
echo "=== Output files ===" | tee -a $LOG
ls -lh $OUT/*.idx $OUT/*.dat 2>/dev/null | tee -a $LOG
