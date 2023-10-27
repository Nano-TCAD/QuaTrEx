#!/bin/sh
foreach size (0 1) 
        echo ${size}
	mpiexec -n 88 -f samples/run_configs/mpd_cpu.run python samples/CNT_biased${size}.py
end


