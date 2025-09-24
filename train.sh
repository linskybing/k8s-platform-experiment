#! /bin/bash

#!/bin/bash

# Simulation parameters
EPOCHS=100
USERS=12

# Array to store elapsed time for each user
declare -A USER_TIME

echo "Starting $USERS users training simultaneously, each for $EPOCHS epochs..."

# Function to train a single user and measure its time
train_user() {
    USER_ID=$1
    echo "User $USER_ID starting training..."

    # Record start time in milliseconds
    START=$(date +%s%3N)

    # Example YOLO training command
    yolo task=detect mode=train data=coco8.yaml model=yolo11n.pt epochs=$EPOCHS device=0 batch=1 &> user${USER_ID}_log.txt

    # Record end time in milliseconds
    END=$(date +%s%3N)
    ELAPSED=$((END - START))

    # Save elapsed time in associative array
    USER_TIME[$USER_ID]=$ELAPSED

    echo "User $USER_ID training completed in $ELAPSED milliseconds"
}

# Launch multiple users simultaneously
for i in $(seq 1 $USERS); do
    train_user $i &
done

# Wait for all users to finish
wait

echo ""
