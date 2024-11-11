# mnistSanitation
Test of data sanitation methods

   The provided Python code is a simple MNIST experiment to attempt to find bad labeled data.
   Since MNIST data is very well vetted, the experiment flips a random subset of samples to
   produce a dataset with corrupted and noisy labels. The experiment then attempts to correct
   these mislabeled samples. This is a binary experiment, so the primary classes are represented
   by the MNIST "1" and "7" samples. Out of Distribution (OOD) data may be added to the simulation
   and is represented by random "0","2","3","4","5","6","8",and "9" characters. This set is
   randomly re-labeled as "1" and "7".

## How to run
Follow the steps in one of the options below to run:

1) You may try modifying fullExperiment.sh

2) Perform the following steps a,b once, and c-h iteratively:

   a) Build corrupted data: python mnistExperiment.py --justGenData ...
      pass proportions for each of: --trainOutliers, --testOutliers, --trainFlips, --testFlips

   b) Build corrupted data table: python readMNistMods.py

   c) Generate M Models (call M times): python mnistExperiment.py --skipTest --test <0:M-1> ... 
      see fullExperiment.sh for options: --trainData, --testData, --trainOutlier, --testOutliers,
      --trainFlips, and --testFlips

   d) Generate results for M models: python mnistExperiment.py --test <0:M-1> -m modelTest...
      see fullExperiment.sh for complete set of options

   e) RANSAC results: python ransacCheck.py

   f) Regenerate results with corrected labels: python mnistExperiment --test <0:M-1> -m modelTest...
      see fullExperiment.sh for complete set of options

   g) Compute label correction stats: python flipRecovery.py

   h) Archive iteration's files for later data mining
      see fullExperiment.sh for full set of files to archive.
