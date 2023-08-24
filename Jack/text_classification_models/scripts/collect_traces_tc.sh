#!/bin/bash

# Usage ./collect_traces.sh

##################### CLI ######################################

# Check CLI args
if [ $# -eq 0 ]; then
	echo -e "Usage: ./collect_traces.sh <Num of samples to collect per app> <Sampling Rate>\n"
	exit 1
fi

if [ "$1" == "-help" ]; then
	echo -e "Usage: ./collect_traces.sh <Num of samples to collect per program> <Sampling Rate> <sc/mc>\n"
	echo -e "	Num of samples to collect per app: The number of times to run each program\n"
	echo -e "	Sampling Rate: Sampling interval for tegrastats in ms\n"
	exit 1
fi

#sudo /usr/bin/jetson_clocks

# Print selected information
echo "Num of samples: $1"
echo "Sampling Rate: ${2}ms"

# List of programs to collect data from
declare -a programs=("deberta-base-mnli" "Text_classification_model_1_pytorch" "finbert" "roberta-hate-speech-dynabench-r4-target")

##################### Check files available ######################################

# Check if all files needed are available

echo -e "\nChecking for necessary files..."
cd ..
[ -d "imagenet" ] && echo "imagenet dataset directory found" || "Error: imagenet directory does not exist."
[ -d "sentences" ] && echo "sentences dataset directory found" || "Error: sentences directory does not exist."

cd models
for i in "${programs[@]}"; do
	if [ -f "$PWD/$i.py" ]; then
		echo -e " $i.py \t\t\t FOUND"
	else
		echo -e "!! $i.py \t\t\t MISSING !!"
		exit 1
	fi
done
cd ../scripts

start_time=$(date +"%r")
echo -e "\nStart time : $start_time"
##################### Collect Traces ######################################

echo -e "\n#### Starting Data Collection ####\n"
sleep 2

# Number of samples (N) to collect per program
N=$1

# Sampling Rate (in ms)
freq=$2

# Command
cmd=(python3)

mkdir raw_data

# Collect N traces from each program
for i in "${programs[@]}"; do
	run_cmd=${cmd[@]}
	run_cmd+=(../models/${i}.py)

	# Program dir to store traces
	mkdir ${i}
	# Run the program for 5 times before data collection
	for j in $(seq 1 5); do
		echo -e "\n##############################################################################"
		echo -e "############ ${i} | Warmup Run: ${j} / 5 #############"
		echo -e "##############################################################################\n"

		${run_cmd[@]} ../imagenet/ ../sentences/
		sleep 4
		echo -e "\n"
	done
	for j in $(seq 1 $N); do
		echo -e "\n##############################################################################"
		echo -e "####### ${i} | Run: ${j} / $N | Sampling Freq: $freq  #######"
		echo -e "##############################################################################\n"
	
		# Start tegrastats monitoring for 2s
		tegrastats --interval ${freq} --logfile ${i}_${freq}ms_${j}.txt &
		# Run command/program
		${run_cmd[@]} ../imagenet/ ../sentences/
		tegrastats --stop
		# Time to let commands finish and cleanup
		sleep 4
		echo -e "\n"
	done
	# Move all traces to appropriate dir
	mv *.txt ${i}
	mv ${i} raw_data
	unset run_cmd[-1]
done

sleep 1s

##################### Finished ######################################

end_time=$(date +"%r")
echo "Start time : $start_time"
echo "End time : $end_time"
echo -e "\n########### ALL RUNS FINISHED ###########\n\n"
