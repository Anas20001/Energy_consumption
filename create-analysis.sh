#!/bin/bash

if [ "$#" -ne 2 ]; then 
    echo "Usage: $0 csv_folder analysis_folder"
    exit 1
fi 

CSV_FOLDER=$1
ANALYSIS_FOLDER=$2


if [ ! -d "$CSV_FOLDER" ]; then 
    echo "Directory $CSV_FOLDER does not exist."
    exit 1
fi 

if [ -z "$(ls -A "$CSV_FOLDER"/*.csv 2>/dev/null)" ]; then
  echo "No .csv files found in the directory $CSV_FOLDER"
  exit 1
fi


ANALYSIS_DIR="$ANALYSIS_FOLDER/Analysis"
mkdir -p $ANALYSIS_DIR

echo "Checking .csv files in $CSV_FOLDER"


for csv_file in $CSV_FOLDER/*.csv; do
    echo "Found csv file: $csv_file"

    
    SUB_DIR="$ANALYSIS_DIR/$(basename "$csv_file" .csv)"
    echo "Creating sub-directory: $SUB_DIR"
    
    mkdir -p $SUB_DIR
done 

echo "Creating placeholder for the csv files in $CSV_FOLDER"
