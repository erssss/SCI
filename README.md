# SCI

This repository is for paper "Win-Win: On Simultaneous Clustering and Imputing over Incomplete Data".

## File Structure

* code: source code of algorithms. For MICE, HC, kPOD, CSDI, kCMM, MForest, GAIN, we use the open source implementations for them, i.e., 
  - MICE: https://scikit-learn.org/stable/modules/generated/sklearn.impute.IterativeImputer.html
  - HC: https://github.com/HoloClean/holoclean
  - CSDI: https://github.com/ermongroup/csdi
  - kPOD: https://github.com/iiradia/kPOD
  - kCMM: https://github.com/clarkdinh/k-cmm
  - MForest: https://hackage.haskell.org/package/MissingPy 
  - GAIN: https://github.com/jsyoon0823/GAIN 
* data: dataset source files of all seven public data collections used in experiments.

## Dataset

* Crx: https://sci2s.ugr.es/keel/dataset.php?cod=59
* Horse: https://sci2s.ugr.es/keel/dataset.php?cod=180
* Soybean: https://archive.ics.uci.edu/dataset/90/soybean+large
* Dermatology: https://sci2s.ugr.es/keel/dataset.php?cod=60
* Banknote: https://archive.ics.uci.edu/dataset/267/banknote+authentication
* Solar Flare: https://archive.ics.uci.edu/dataset/89/solar+flare

## Dependencies

python 3.9

```
gurobipy==10.0.3
missingpy==0.2.0
numpy==1.23.5
pandas==2.1.3
scikit_learn==0.24.0
scipy==1.11.4
```

## Instruction

``` sh
cd code
python main.py
```