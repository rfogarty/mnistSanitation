#!/bin/bash

export TESTNUM=5
export MAX_SIM=39
export OODPROP=0.01
export FLIPPROP=0.25
export TRAINDATA="TrainingFlipsTest${TESTNUM}_${OODPROP}_${OODPROP}_${FLIPPROP}_${FLIPPROP}.pkl"
export TESTDATA="TestingFlipsTest${TESTNUM}_${OODPROP}_${OODPROP}_${FLIPPROP}_${FLIPPROP}.pkl"
export TRAINRES="test${TESTNUM}Training.csv"
export TESTRES="test${TESTNUM}Testing.csv"

echo "############################ Generate Data #################################"
python mnistExperiment.py --test $TESTNUM \
                                    --trainOutliers $OODPROP --testOutliers $OODPROP --trainFlips $FLIPPROP --testFlips $FLIPPROP \
                                    --justGenData

echo "################# Parse out flipped indices for convenience ################"
python readMNistMods.py

echo "########################### Generate Models ################################"
echo $(seq 0 $MAX_SIM) | xargs -n1 echo | parallel -j10 -IRPLC python mnistExperiment.py --test RPLC \
                                                                                         --trainData $TRAINDATA --testData $TESTDATA \
                                                                                         --trainOutliers $OODPROP --testOutliers $OODPROP --trainFlips $FLIPPROP --testFlips $FLIPPROP \
                                                                                         --trainingLimitRatio 0.8 --skipTest \
                                                                                         | tee "train${OODPROP}OOD_Flip${FLIPPROP}_${TESTNUM}.txt"


echo "########################### Generate Results ##############################"
echo $(seq 0 $MAX_SIM) | xargs -n1 echo | xargs -n1 -IRPLC python mnistExperiment.py --test RPLC -m "modelTestRPLC_${OODPROP}_${OODPROP}_${FLIPPROP}_${FLIPPROP}.ptm" \
                                                                                     --trainData $TRAINDATA \
                                                                                     --testData $TESTDATA \
                                                                                     --trainOutliers $OODPROP --testOutliers $OODPROP --trainFlips $FLIPPROP --testFlips $FLIPPROP \
                                                                                     --trainResults $TRAINRES --testResults $TESTRES \
                                                                                     | tee "train${OODPROP}OOD_Flip${FLIPPROP}_${TESTNUM}_ransac.txt"

echo "########### Estimate improperly labeled samples, and difficult (possibly OOD) samples ########"
python ransacCheck.py

echo "################## Regenerate Results with flipped samples ####################"
echo $(seq 0 $MAX_SIM) | xargs -n1 echo | xargs -n1 -IRPLC python mnistExperiment.py --test RPLC -m "modelTestRPLC_${OODPROP}_${OODPROP}_${FLIPPROP}_${FLIPPROP}.ptm" \
                                                                                     --trainData $TRAINDATA \
                                                                                     --testData $TESTDATA \
                                                                                     --trainOutliers $OODPROP --testOutliers $OODPROP --trainFlips $FLIPPROP --testFlips $FLIPPROP \
                                                                                     --trainResults "test${TESTNUM}TrainingFlipped.csv" --testResults "test${TESTNUM}TestingFlipped.csv" \
                                                                                     --flipTraining questionableTraining.csv --flipTesting questionableTesting.csv \
                                                                                     | tee "train${OODPROP}OOD_Flip${FLIPPROP}_${TESTNUM}_flipped.txt"


echo "############# Finally estimate statistics on recovered samples ###############"
# TODO: rewite these to pass in appropriate file names
python flipRecovery.py

echo "###################### Refinement (Round 2) ###########################"
mv -v oodTesting.csv Round0_oodTesting.csv
mv -v oodTraining.csv Round0_oodTraining.csv
mv -v questionableTesting.csv Round0_questionableTesting.csv
mv -v $TRAINRES "Round0_${TRAINRES}"
mv -v $TESTRES "Round0_${TESTRES}"
mv -v test${TESTNUM}TestingFlipped.csv  Round0_test${TESTNUM}TestingFlipped.csv
mv -v test${TESTNUM}TrainingFlipped.csv Round0_test${TESTNUM}TrainingFlipped.csv
echo $(seq 0 $MAX_SIM) | xargs -n1 echo | xargs -IRPLC mv -v modelTestRPLC_${OODPROP}_${OODPROP}_${FLIPPROP}_${FLIPPROP}.ptm Round0_modelTestRPLC_${OODPROP}_${OODPROP}_${FLIPPROP}_${FLIPPROP}.ptm

echo "########################### Generate Models ################################"
echo $(seq 0 $MAX_SIM) | xargs -n1 echo | parallel -j10 -IRPLC python mnistExperiment.py --test RPLC \
                                                                                         --trainData $TRAINDATA --testData $TESTDATA \
                                                                                         --trainOutliers $OODPROP --testOutliers $OODPROP --trainFlips $FLIPPROP --testFlips $FLIPPROP \
                                                                                         --flipTraining questionableTraining.csv \
                                                                                         --trainingLimitRatio 0.8 --skipTest \
                                                                                         | tee "train${OODPROP}OOD_Flip${FLIPPROP}_${TESTNUM}_r2.txt"

mv -v questionableTraining.csv Round0_questionableTraining.csv

echo "########################### Generate Results ##############################"
echo $(seq 0 $MAX_SIM) | xargs -n1 echo | xargs -n1 -IRPLC python mnistExperiment.py --test RPLC -m "modelTestRPLC_${OODPROP}_${OODPROP}_${FLIPPROP}_${FLIPPROP}.ptm" \
                                                                                     --trainData $TRAINDATA \
                                                                                     --testData $TESTDATA \
                                                                                     --trainOutliers $OODPROP --testOutliers $OODPROP --trainFlips $FLIPPROP --testFlips $FLIPPROP \
                                                                                     --trainResults $TRAINRES --testResults $TESTRES \
                                                                                     | tee "train${OODPROP}OOD_Flip${FLIPPROP}_${TESTNUM}_ransac_r2.txt"

echo "########### Estimate improperly labeled samples, and difficult (possibly OOD) samples ########"
python ransacCheck.py

echo "################## Regenerate Results with flipped samples ####################"
echo $(seq 0 $MAX_SIM) | xargs -n1 echo | xargs -n1 -IRPLC python mnistExperiment.py --test RPLC -m "modelTestRPLC_${OODPROP}_${OODPROP}_${FLIPPROP}_${FLIPPROP}.ptm" \
                                                                                     --trainData $TRAINDATA \
                                                                                     --testData $TESTDATA \
                                                                                     --trainOutliers $OODPROP --testOutliers $OODPROP --trainFlips $FLIPPROP --testFlips $FLIPPROP \
                                                                                     --trainResults "test${TESTNUM}TrainingFlipped.csv" --testResults "test${TESTNUM}TestingFlipped.csv" \
                                                                                     --flipTraining questionableTraining.csv --flipTesting questionableTesting.csv \
                                                                                     | tee "train${OODPROP}OOD_Flip${FLIPPROP}_${TESTNUM}_flipped_r2.txt"


echo "############# Finally estimate statistics on recovered samples ###############"
# TODO: rewite these to pass in appropriate file names
python flipRecovery.py


echo "###################### Refinement (Round 3) w/ RANSAC ###########################"
mv -v oodTesting.csv Round1_oodTesting.csv
mv -v questionableTesting.csv Round1_questionableTesting.csv
mv -v $TRAINRES "Round1_${TRAINRES}"
mv -v $TESTRES "Round1_${TESTRES}"
mv -v test${TESTNUM}TestingFlipped.csv  Round1_test${TESTNUM}TestingFlipped.csv
mv -v test${TESTNUM}TrainingFlipped.csv Round1_test${TESTNUM}TrainingFlipped.csv
echo $(seq 0 $MAX_SIM) | xargs -n1 echo | xargs -IRPLC mv -v modelTestRPLC_${OODPROP}_${OODPROP}_${FLIPPROP}_${FLIPPROP}.ptm Round1_modelTestRPLC_${OODPROP}_${OODPROP}_${FLIPPROP}_${FLIPPROP}.ptm

echo "########################### Generate Models ################################"
echo $(seq 0 $MAX_SIM) | xargs -n1 echo | parallel -j10 -IRPLC python mnistExperiment.py --test RPLC \
                                                                                         --trainData $TRAINDATA --testData $TESTDATA \
                                                                                         --trainOutliers $OODPROP --testOutliers $OODPROP --trainFlips $FLIPPROP --testFlips $FLIPPROP \
                                                                                         --flipTraining questionableTraining.csv --blockTraining oodTraining.csv \
                                                                                         --trainingLimitRatio 0.8 --skipTest \
                                                                                         | tee "train${OODPROP}OOD_Flip${FLIPPROP}_${TESTNUM}_r3.txt"

mv -v questionableTraining.csv Round1_questionableTraining.csv
mv -v oodTraining.csv Round1_oodTraining.csv

echo "########################### Generate Results ##############################"
echo $(seq 0 $MAX_SIM) | xargs -n1 echo | xargs -n1 -IRPLC python mnistExperiment.py --test RPLC -m "modelTestRPLC_${OODPROP}_${OODPROP}_${FLIPPROP}_${FLIPPROP}.ptm" \
                                                                                     --trainData $TRAINDATA \
                                                                                     --testData $TESTDATA \
                                                                                     --trainOutliers $OODPROP --testOutliers $OODPROP --trainFlips $FLIPPROP --testFlips $FLIPPROP \
                                                                                     --trainResults $TRAINRES --testResults $TESTRES \
                                                                                     | tee "train${OODPROP}OOD_Flip${FLIPPROP}_${TESTNUM}_ransac_r3.txt"

echo "########### Estimate improperly labeled samples, and difficult (possibly OOD) samples ########"
python ransacCheck.py

echo "################## Regenerate Results with flipped samples ####################"
echo $(seq 0 $MAX_SIM) | xargs -n1 echo | xargs -n1 -IRPLC python mnistExperiment.py --test RPLC -m "modelTestRPLC_${OODPROP}_${OODPROP}_${FLIPPROP}_${FLIPPROP}.ptm" \
                                                                                     --trainData $TRAINDATA \
                                                                                     --testData $TESTDATA \
                                                                                     --trainOutliers $OODPROP --testOutliers $OODPROP --trainFlips $FLIPPROP --testFlips $FLIPPROP \
                                                                                     --trainResults "test${TESTNUM}TrainingFlipped.csv" --testResults "test${TESTNUM}TestingFlipped.csv" \
                                                                                     --flipTraining questionableTraining.csv --flipTesting questionableTesting.csv \
                                                                                     | tee "train${OODPROP}OOD_Flip${FLIPPROP}_${TESTNUM}_flipped_r3.txt"


echo "############# Finally estimate statistics on recovered samples ###############"
# TODO: rewite these to pass in appropriate file names
python flipRecovery.py

