#!/bin/bash

# Usage ./collect_traces.sh

##################### CLI ######################################

# Check CLI args
if [ $# -eq 0 ]; then
	echo -e "Usage: ./parse_data.sh <Data directory>\n"
	exit 1
fi

if [ "$1" == "-help" ]; then
	echo -e "Usage: ./parse_data.sh <Data directory><sc/mc>\n"
	exit 1
fi

#sudo /usr/bin/jetson_clocks

# Print selected information
echo "Parsing data from directory: $1"


start_time=$(date +"%r")
echo -e "\nStart time : $start_time"
##################### Collect Traces ######################################

echo -e "\n#### Start Parsing ####\n"
sleep 2

# Number of samples (N) to collect per program
raw_data=$1

# Command
cmd=(python3)

cp -R ${raw_data} parsed_data
cd parsed_data
# Collect N traces from each program
for dir in */; do

	# Program dir to store traces
	cd ${dir}
	for file in *.txt; do
		echo -e "\n##############################################################################"
		echo -e "####### Parsing File: ${file}  #######"
		echo -e "##############################################################################\n"
	
		# Run command/program
		python ../../../tegrastats_parser-main/main.py --only_parse --interval 1 --log_file "${file}"
		sleep 4
		echo -e "\n"
	done
	# Delete all .txt files
	rm *.txt
	cd ..
done

cd ..
sleep 1s

##################### Finished ######################################

end_time=$(date +"%r")
echo "Start time : $start_time"
echo "End time : $end_time"
echo -e "\n########### ALL RUNS FINISHED ###########\n\n"
