import pandas as pd
import numpy as np

# Note: MNIST 13007 1/7s train, 2163 1/7s test
def testCommon(origFile,recFile,dataset='',numIID=-1) :
    dfo = pd.read_csv(origFile,index_col=False)
    dfr = pd.read_csv(recFile,index_col=False)

    ao = np.array(dfo['Flipped'])
    ar = np.array(dfr['Index'])

    common = [f for f in ao if f in ar]
    missed = [f for f in ao if f not in ar]
    oods = [f for f in ar if f >= numIID]
    falselyDet = [f1 for f1 in [f0 for f0 in ar if f0 < numIID] if f1 not in ao]
    df_oods = pd.DataFrame({'Index':oods})
    df_oods.to_csv(f'{dataset}OODflips.csv',index=False)
    df_FAs = pd.DataFrame({'Index':falselyDet})
    df_FAs.to_csv(f'{dataset}FAflips.csv',index=False)
    df_TP = pd.DataFrame({'Index':common})
    df_TP.to_csv(f'{dataset}TPflips.csv',index=False)
    df_FN = pd.DataFrame({'Index':missed})
    df_FN.to_csv(f'{dataset}FNflips.csv',index=False)
    #breakpoint()
    commonOrOOD = common.copy()
    commonOrOOD.extend(oods)
    df_improved = pd.DataFrame({'Index':commonOrOOD})
    df_improved.to_csv(f'{dataset}Improvedflips.csv',index=False)
    totalError = missed.copy()
    totalError.extend(falselyDet)
    df_degraded = pd.DataFrame({'Index':totalError})
    df_degraded.to_csv(f'{dataset}Degradedflips.csv',index=False)

    
    print(f'Dataset {dataset} |IID flipped|={len(ao)}, |Detected|={len(ar)}, |Recovered|={len(common)}, |Missed|={len(ao)-len(common)}, |Defamed|={len(falselyDet)}={len(ar)-len(common)-len(oods)}, |OOD|={len(oods)}')
    return (len(common)/len(ao),(len(ar)-len(common))/len(ar),(len(ar)-len(common)-len(oods))/len(ar))
    
    
detectionRate,falseAlarmRate,compFAR = testCommon('OriginalTrainingFlips.csv','questionableTraining.csv',dataset='training',numIID=13007)
print(f'Training detectionRate={detectionRate}, falseAlarmRate={falseAlarmRate}, compFAR={compFAR}')

detectionRate,falseAlarmRate,compFAR = testCommon('OriginalTestingFlips.csv','questionableTesting.csv',dataset='testing',numIID=2163)
print(f'Testing detectionRate={detectionRate}, falseAlarmRate={falseAlarmRate}, compFAR={compFAR}')

