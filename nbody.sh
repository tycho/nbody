#!/bin/bash
set -ex

function run_nbody {
	/work/nbody --verbose "$@"
}
function run_nbody_gpu {
	run_nbody --no-cpu --iterations 1 --cycle-after 6 --bodies $1
}
function run_nbody_crosscheck {
	run_nbody --iterations 1 --cycle-after 4 --bodies $1
}

# Environment information
nvidia-smi

# Ensure the build is sane, we should see that CPU/GPU numbers match up.
run_nbody_crosscheck 32

# Now simulate varying universe sizes
run_nbody_gpu 128
run_nbody_gpu 256
run_nbody_gpu 512
run_nbody_gpu 1024
