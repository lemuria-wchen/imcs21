# DialoAMC

This repo contains a new corpus benchmark called **DialoAMC** for automated medical consultation system, as well as the code for reproducing the experiments.

### News

- The test set of DialoAMC is host on [CBLEU](https://tianchi.aliyun.com/specials/promotion/2021chinesemedicalnlpleaderboardchallenge) at [TIANCHI](https://tianchi.aliyun.com/competition/gameList/activeList) platform. See more details in https://github.com/lemuria-wchen/imcs21-cblue. Welcome to submit your results on [CBLEU](https://tianchi.aliyun.com/specials/promotion/2021chinesemedicalnlpleaderboardchallenge), or compare our results on the validation set.   
- Please see more details in our arxiv paper [A Benchmark for Automatic Medical Consultation System: Frameworks, Tasks and Datasets]().
- DialoAMC is released, containing a total of 4,116 annotated medical consultation records that covers 10 pediatric diseases.   

### Tasks

- NER: [BERT-CRF](https://github.com/lemuria-wchen/imcs21/tree/main/task/NER/BERT-NER), [LEBERT](https://github.com/lemuria-wchen/imcs21/tree/main/task/NER/LEBERT-NER) 
- DAC: [TextCNN](https://github.com/lemuria-wchen/imcs21/tree/main/task/DAC/DNN-DAC), [TextRNN](https://github.com/lemuria-wchen/imcs21/tree/main/task/DAC/DNN-DAC), [TextRCNN](https://github.com/lemuria-wchen/imcs21/tree/main/task/DAC/DNN-DAC), [DPCNN](https://github.com/lemuria-wchen/imcs21/tree/main/task/DAC/DNN-DAC), [BERT](https://github.com/lemuria-wchen/imcs21/tree/main/task/DAC/BERT-DAC), [ERNIE](https://github.com/lemuria-wchen/imcs21/tree/main/task/DAC/BERT-DAC)
- SRI: [BERT-MLC](https://github.com/lemuria-wchen/imcs21/tree/main/task/SLI/MLC-SLI), [BERT-MTL](https://github.com/lemuria-wchen/imcs21/tree/main/task/SLI/MTL-SLI)
- MRG: [Seq2seq](https://github.com/lemuria-wchen/imcs21/tree/main/task/MRG/opennmt), [PG](https://github.com/lemuria-wchen/imcs21/tree/main/task/MRG/opennmt), [Transformer](https://github.com/lemuria-wchen/imcs21/tree/main/task/MRG/opennmt), [ProphetNet](https://github.com/lemuria-wchen/imcs21/tree/main/task/MRG/prophetnet), [T5](https://github.com/lemuria-wchen/imcs21/tree/main/task/MRG/t5)
- DDP: [DQN](https://github.com/lemuria-wchen/imcs21/tree/main/task/DDP), [KQ-DQN](https://github.com/lemuria-wchen/imcs21/tree/main/task/DDP), [REFUEL](https://github.com/lemuria-wchen/imcs21/tree/main/task/DDP), [GAMP](https://github.com/lemuria-wchen/imcs21/tree/main/task/DDP), [HRL](https://github.com/lemuria-wchen/imcs21/tree/main/task/DDP)

### Results of NER Task

<table>
<thead>
  <tr>
    <th rowspan="2">Models</th>
    <th rowspan="2">Split</th>
    <th colspan="6">Entity-Level</th>
    <th colspan="3">Token-Level</th>
  </tr>
  <tr>
    <th>SX</th>
    <th>DN</th>
    <th>DC</th>
    <th>EX</th>
    <th>OP</th>
    <th>Overall</th>
    <th>P</th>
    <th>R</th>
    <th>F1</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td rowspan="2">Lattice LSTM</td>
    <td>Dev</td>
    <td>90.61</td>
    <td>88.12</td>
    <td>90.89</td>
    <td>90.44</td>
    <td>91.14</td>
    <td>90.33</td>
    <td>89.62</td>
    <td>91.00</td>
    <td>90.31</td>
  </tr>
  <tr>
    <td>Test</td>
    <td>90.00</td>
    <td>87.84</td>
    <td>91.32</td>
    <td>90.55</td>
    <td>93.42</td>
    <td>90.10</td>
    <td>89.37</td>
    <td>90.84</td>
    <td>90.10</td>
  </tr>
  <tr>
    <td rowspan="2">BERT</td>
    <td>Dev</td>
    <td>91.15</td>
    <td>89.74</td>
    <td>90.97</td>
    <td>90.74</td>
    <td>92.57</td>
    <td>90.95</td>
    <td>88.99</td>
    <td>92.43</td>
    <td>90.68</td>
  </tr>
  <tr>
    <td>Test</td>
    <td>90.59</td>
    <td>89.97</td>
    <td>90.54</td>
    <td>90.48</td>
    <td>94.39</td>
    <td>90.64</td>
    <td>88.46</td>
    <td>92.35</td>
    <td>90.37</td>
  </tr>
  <tr>
    <td rowspan="2">ERNIE</td>
    <td>Dev</td>
    <td>91.28</td>
    <td>89.68</td>
    <td>90.92</td>
    <td>91.15</td>
    <td>92.65</td>
    <td>91.08</td>
    <td>89.36</td>
    <td>92.46</td>
    <td>90.88</td>
  </tr>
  <tr>
    <td>Test</td>
    <td>90.67</td>
    <td>89.89</td>
    <td>90.73</td>
    <td>90.97</td>
    <td>94.33</td>
    <td>90.78</td>
    <td>88.87</td>
    <td>92.27</td>
    <td>90.53</td>
  </tr>
  <tr>
    <td rowspan="2">FLAT</td>
    <td>Dev</td>
    <td>90.90</td>
    <td>89.95</td>
    <td>90.64</td>
    <td>90.58</td>
    <td>93.14</td>
    <td>90.80</td>
    <td>88.89</td>
    <td>92.23</td>
    <td>90.53</td>
  </tr>
  <tr>
    <td>Test</td>
    <td>90.45</td>
    <td>89.67</td>
    <td>90.35</td>
    <td>91.12</td>
    <td>93.47</td>
    <td>90.58</td>
    <td>88.76</td>
    <td>92.07</td>
    <td>90.38</td>
  </tr>
  <tr>
    <td rowspan="2">LEBERT</td>
    <td>Dev</td>
    <td>92.61</td>
    <td>90.67</td>
    <td>90.71</td>
    <td>92.39</td>
    <td>92.30</td>
    <td>92.11</td>
    <td>86.95</td>
    <td>93.05</td>
    <td>89.90</td>
  </tr>
  <tr>
    <td>Test</td>
    <td>92.14</td>
    <td>90.31</td>
    <td>91.16</td>
    <td>92.35</td>
    <td>93.94</td>
    <td>91.92</td>
    <td>86.53</td>
    <td>92.91</td>
    <td>89.60</td>
  </tr>
</tbody>
</table>

### Results of DAC Task

<table>
<thead>
  <tr>
    <th>Models</th>
    <th>Split</th>
    <th>P</th>
    <th>R</th>
    <th>F1</th>
    <th>Acc</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td rowspan="2">TextCNN</td>
    <td>Dev</td>
    <td>73.09</td>
    <td>70.26</td>
    <td>71.26</td>
    <td>77.77</td>
  </tr>
  <tr>
    <td>Test</td>
    <td>74.76</td>
    <td>70.06</td>
    <td>71.91</td>
    <td>78.93</td>
  </tr>
  <tr>
    <td rowspan="2">TextRNN</td>
    <td>Dev</td>
    <td>74.02</td>
    <td>68.43</td>
    <td>70.71</td>
    <td>78.14</td>
  </tr>
  <tr>
    <td>Test</td>
    <td>72.53</td>
    <td>70.99</td>
    <td>71.23</td>
    <td>78.46</td>
  </tr>
  <tr>
    <td rowspan="2">TextRCNN</td>
    <td>Dev</td>
    <td>71.43</td>
    <td>72.68</td>
    <td>71.50</td>
    <td>77.67</td>
  </tr>
  <tr>
    <td>Test</td>
    <td>74.29</td>
    <td>72.07</td>
    <td>72.84</td>
    <td>79.49</td>
  </tr>
  <tr>
    <td rowspan="2">DPCNN</td>
    <td>Dev</td>
    <td>70.10</td>
    <td>70.91</td>
    <td>69.85</td>
    <td>77.14</td>
  </tr>
  <tr>
    <td>Test</td>
    <td>71.29</td>
    <td>71.82</td>
    <td>71.38</td>
    <td>77.91</td>
  </tr>
  <tr>
    <td rowspan="2">BERT</td>
    <td>Dev</td>
    <td>75.19</td>
    <td>76.31</td>
    <td>75.66</td>
    <td>81.00</td>
  </tr>
  <tr>
    <td>Test</td>
    <td>75.53</td>
    <td>77.24</td>
    <td>76.28</td>
    <td>81.65</td>
  </tr>
  <tr>
    <td rowspan="2">ERNIE</td>
    <td>Dev</td>
    <td>76.04</td>
    <td>76.82</td>
    <td>76.37</td>
    <td>81.60</td>
  </tr>
  <tr>
    <td>Test</td>
    <td>75.72</td>
    <td>76.94</td>
    <td>76.25</td>
    <td>81.91</td>
  </tr>
</tbody>
</table>

#### Confusion matrix of ERNIE model on test set 

<img src="figures/da_ernie_confusion.png" width="400" height="300" alt="Confusion" align=center/>

### Results of SLI Task

<table>
<thead>
  <tr>
    <th rowspan="2">Models</th>
    <th rowspan="2">Split</th>
    <th colspan="3">Example-based</th>
    <th colspan="3">Label-based</th>
  </tr>
  <tr>
    <th>SA</th>
    <th>HL</th>
    <th>HS</th>
    <th>P</th>
    <th>R</th>
    <th>F1</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td colspan="8">SLI-EXP (Symptom&nbsp;&nbsp;&nbsp;Recognition)</td>
  </tr>
  <tr>
    <td rowspan="2">BERT-MLC</td>
    <td>Dev</td>
    <td>75.63</td>
    <td>10.12</td>
    <td>86.53</td>
    <td>86.50</td>
    <td>93.80</td>
    <td>90.00</td>
  </tr>
  <tr>
    <td>Test</td>
    <td>73.24</td>
    <td>10.10</td>
    <td>84.58</td>
    <td>86.33</td>
    <td>93.14</td>
    <td>89.60</td>
  </tr>
  <tr>
    <td colspan="8">SLI-IMP (Symptom Recognition)</td>
  </tr>
  <tr>
    <td rowspan="2">BERT-MLC</td>
    <td>Dev</td>
    <td>37.58</td>
    <td>38.44</td>
    <td>83.03</td>
    <td>85.03</td>
    <td>95.40</td>
    <td>89.91</td>
  </tr>
  <tr>
    <td>Test</td>
    <td>36.62</td>
    <td>37.36</td>
    <td>83.12</td>
    <td>86.13</td>
    <td>94.80</td>
    <td>90.26</td>
  </tr>
  <tr>
    <td rowspan="2">BERT-MTL</td>
    <td>Dev</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td>Test</td>
    <td>40.19</td>
    <td>31.89</td>
    <td>85.26</td>
    <td>89.34</td>
    <td>94.49</td>
    <td>91.84</td>
  </tr>
  <tr>
    <td colspan="8">SLI-IMP (Symptom Inference)</td>
  </tr>
  <tr>
    <td></td>
    <td></td>
    <td>POS</td>
    <td>NEG</td>
    <td>NS</td>
    <td>Overall</td>
    <td>Acc</td>
    <td></td>
  </tr>
  <tr>
    <td rowspan="2">BERT-MLC</td>
    <td>Dev</td>
    <td>80.69</td>
    <td>43.83</td>
    <td>56.63</td>
    <td>60.39</td>
    <td>71.17</td>
    <td></td>
  </tr>
  <tr>
    <td>Test</td>
    <td>80.54</td>
    <td>42.98</td>
    <td>56.06</td>
    <td>59.86</td>
    <td>70.64</td>
    <td></td>
  </tr>
  <tr>
    <td rowspan="2">BERT-MTL</td>
    <td>Dev</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
    <td></td>
  </tr>
  <tr>
    <td>Test</td>
    <td>81.31</td>
    <td>50.77</td>
    <td>63.32</td>
    <td>-</td>
    <td>-</td>
    <td></td>
  </tr>
</tbody>
</table>

### Results of MRG Task

<table>
<thead>
  <tr>
    <th>Models</th>
    <th>B-2</th>
    <th>B-4</th>
    <th>R-1</th>
    <th>R-2</th>
    <th>R-L</th>
    <th>C-F1</th>
    <th>D-Acc</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>Seq2Seq</td>
    <td>54.43</td>
    <td>43.95</td>
    <td>54.13</td>
    <td>43.98</td>
    <td>50.42</td>
    <td>36.73</td>
    <td>48.34</td>
  </tr>
  <tr>
    <td>PG</td>
    <td>58.31</td>
    <td>49.31</td>
    <td>59.46</td>
    <td>49.79</td>
    <td>56.34</td>
    <td>46.36</td>
    <td>56.60</td>
  </tr>
  <tr>
    <td>Transformer</td>
    <td>58.57</td>
    <td>47.67</td>
    <td>57.25</td>
    <td>46.29</td>
    <td>53.29</td>
    <td>40.64</td>
    <td>54.50</td>
  </tr>
  <tr>
    <td>T5</td>
    <td>62.57</td>
    <td>52.48</td>
    <td>61.20</td>
    <td>50.98</td>
    <td>58.18</td>
    <td>46.55</td>
    <td>47.60</td>
  </tr>
  <tr>
    <td>ProphetNet</td>
    <td>58.11</td>
    <td>49.06</td>
    <td>61.18</td>
    <td>50.33</td>
    <td>57.94</td>
    <td>49.61</td>
    <td>55.36</td>
  </tr>
</tbody>
</table>

### Results of DDP Task

<table>
<thead>
  <tr>
    <th>Models</th>
    <th>Recall</th>
    <th>Acc</th>
    <th># Turns</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>DQN</td>
    <td>0.047</td>
    <td>0.408</td>
    <td>9.750</td>
  </tr>
  <tr>
    <td>REFUEL</td>
    <td>0.262</td>
    <td>0.505</td>
    <td>5.500</td>
  </tr>
  <tr>
    <td>KR-DQN</td>
    <td>0.279</td>
    <td>0.485</td>
    <td>5.950</td>
  </tr>
  <tr>
    <td>GAMP</td>
    <td>0.067</td>
    <td>0.500</td>
    <td>1.780</td>
  </tr>
  <tr>
    <td>HRL</td>
    <td>0.295</td>
    <td>0.556</td>
    <td>6.990</td>
  </tr>
</tbody>
</table>

### How to Cite

If you extend or use this work, please cite the [paper]() where it was introduced. 
