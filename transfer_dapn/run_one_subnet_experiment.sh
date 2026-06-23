#!/usr/bin/env bash
# p1 variant for fixed_dmz_one_subnet_4host — delegates to run_p1_experiment.sh
SCENARIO=one_subnet exec "$(dirname "$0")/run_p1_experiment.sh"
