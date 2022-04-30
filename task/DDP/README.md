## Task 5：Diagnosis-oriented Dialogue Policy (DDP)

This dir contains the code of **DQN, KQ-DQN, REFUEL, GAMP, HRL** model for DDP task.

The code is copied from https://github.com/Guardianzc/DISCOpen-MedBox-DialoDiagnosis.

### Requirements

```shell
pip install OpenMedicalChatBox
```

### Preprocess

- generate goal.set / goal_test_set
 
```shell
python preprocess.py
```

- generate slot_set / disease_set / disease_symptom /  

```shell
python data+generator.py
```

### Train

```shell
python demo.py
```
