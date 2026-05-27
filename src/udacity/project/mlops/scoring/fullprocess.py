import training
import scoring
import deployment
import diagnostics
import reporting

################## Check and read new data
# First, read ingestedfiles.txt

# Second, determine whether the source data folder has files that aren't listed
# in ingestedfiles.txt



################## Deciding whether to proceed, part 1
# If you found new data, you should proceed. otherwise, do end the process here


################## Checking for model drift
# Check whether the score from the deployed model is different from the score 
# from the model that uses the newest ingested data


################## Deciding whether to proceed, part 2
# If you found model drift, you should proceed. otherwise, do end the process here



################## Re-deployment
# If you found evidence for model drift, re-run the deployment.py script

################## Diagnostics and reporting
# Run diagnostics.py and reporting.py for the re-deployed model







