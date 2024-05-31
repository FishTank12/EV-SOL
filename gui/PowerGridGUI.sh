#!/bin/bash
# PowerGridGUI.sh

# Set the path to your JAR file
JAR_PATH="/home/sari-itani/Documents/GitHub/EV-SOL/gui/PowerGridGUI.jar"
JSON_LIB_PATH="/home/sari-itani/Documents/GitHub/EV-SOL/libs/json-20230618.jar"

# Execute the Java application
java -cp $JAR_PATH:$JSON_LIB_PATH PowerGridGUI
