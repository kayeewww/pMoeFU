import pandas as pd

str = \
    """Train:
Train Loss=1.99578251, Train acc=0.24444444
Global Federated Learning epoch = 0
Test Loss=2.9406, Test accuracy=0.1380
Train:
Train Loss=1.73234931, Train acc=0.30888889
Global Federated Learning epoch = 1
Test Loss=2.6349, Test accuracy=0.1992
Train:
Train Loss=1.81185755, Train acc=0.34222222
Global Federated Learning epoch = 2
Test Loss=2.6212, Test accuracy=0.2374
Train:
Train Loss=1.80710834, Train acc=0.34000000
Global Federated Learning epoch = 3
Test Loss=2.4616, Test accuracy=0.2657
Train:
Train Loss=1.85957025, Train acc=0.33333333
Global Federated Learning epoch = 4
Test Loss=2.2873, Test accuracy=0.3344
Train:
Train Loss=1.95571925, Train acc=0.35111111
Global Federated Learning epoch = 5
Test Loss=2.3914, Test accuracy=0.3202
Train:
Train Loss=1.67056122, Train acc=0.34444444
Global Federated Learning epoch = 6
Test Loss=2.3078, Test accuracy=0.3498
Train:
Train Loss=1.55195235, Train acc=0.44222222
Global Federated Learning epoch = 7
Test Loss=2.2970, Test accuracy=0.3661
Train:
Train Loss=1.60380460, Train acc=0.40666667
Global Federated Learning epoch = 8
Test Loss=2.2328, Test accuracy=0.3847
Train:
Train Loss=1.64237094, Train acc=0.41777778
Global Federated Learning epoch = 9
Test Loss=2.2731, Test accuracy=0.3790
Train:
Train Loss=1.46214971, Train acc=0.45111111
Global Federated Learning epoch = 10
Test Loss=2.2483, Test accuracy=0.4032
Train:
Train Loss=1.44012763, Train acc=0.44666667
Global Federated Learning epoch = 11
Test Loss=2.2414, Test accuracy=0.3998
Train:
Train Loss=1.50805274, Train acc=0.46666667
Global Federated Learning epoch = 12
Test Loss=2.2331, Test accuracy=0.4165
Train:
Train Loss=1.48077585, Train acc=0.47555556
Global Federated Learning epoch = 13
Test Loss=2.2118, Test accuracy=0.4176
Train:
Train Loss=1.71130829, Train acc=0.42888889
Global Federated Learning epoch = 14
Test Loss=2.1755, Test accuracy=0.4349
Train:
Train Loss=1.45053324, Train acc=0.40666667
Global Federated Learning epoch = 15
Test Loss=2.2055, Test accuracy=0.4173
Train:
Train Loss=1.51654129, Train acc=0.42000000
Global Federated Learning epoch = 16
Test Loss=2.1803, Test accuracy=0.4440
Train:
Train Loss=1.43535271, Train acc=0.46888889
Global Federated Learning epoch = 17
Test Loss=2.1797, Test accuracy=0.4427
Train:
Train Loss=1.64333737, Train acc=0.42888889
Global Federated Learning epoch = 18
Test Loss=2.1867, Test accuracy=0.4390
Train:
Train Loss=1.45277213, Train acc=0.45777778
Global Federated Learning epoch = 19
Test Loss=2.1533, Test accuracy=0.4405
Train:
Train Loss=1.29250911, Train acc=0.54888889
Global Federated Learning epoch = 20
Test Loss=2.1286, Test accuracy=0.4665
Train:
Train Loss=1.37841620, Train acc=0.50222222
Global Federated Learning epoch = 21
Test Loss=2.1238, Test accuracy=0.4708
Train:
Train Loss=1.18553130, Train acc=0.55555556
Global Federated Learning epoch = 22
Test Loss=2.0890, Test accuracy=0.4844
Train:
Train Loss=1.43777031, Train acc=0.42444444
Global Federated Learning epoch = 23
Test Loss=2.1337, Test accuracy=0.4748
Train:
Train Loss=1.31519373, Train acc=0.48666667
Global Federated Learning epoch = 24
Test Loss=1.9983, Test accuracy=0.4944
Train:
Train Loss=1.29175805, Train acc=0.59333333
Global Federated Learning epoch = 25
Test Loss=2.0573, Test accuracy=0.4921
Train:
Train Loss=1.49865603, Train acc=0.45111111
Global Federated Learning epoch = 26
Test Loss=2.0579, Test accuracy=0.4933
Train:
Train Loss=1.23283531, Train acc=0.56444444
Global Federated Learning epoch = 27
Test Loss=2.0528, Test accuracy=0.4954
Train:
Train Loss=1.26537284, Train acc=0.57333333
Global Federated Learning epoch = 28
Test Loss=2.0731, Test accuracy=0.4984
Train:
Train Loss=1.12627809, Train acc=0.55555556
Global Federated Learning epoch = 29
Test Loss=1.9870, Test accuracy=0.5244
Train:
Train Loss=1.20430826, Train acc=0.56222222
Global Federated Learning epoch = 30
Test Loss=2.0573, Test accuracy=0.5006
Train:
Train Loss=1.29296830, Train acc=0.51111111
Global Federated Learning epoch = 31
Test Loss=2.0139, Test accuracy=0.5172
Train:
Train Loss=1.36445639, Train acc=0.52666667
Global Federated Learning epoch = 32
Test Loss=1.9856, Test accuracy=0.5324
Train:
Train Loss=1.29209395, Train acc=0.58222222
Global Federated Learning epoch = 33
Test Loss=2.0024, Test accuracy=0.5284
Train:
Train Loss=1.53692655, Train acc=0.51555556
Global Federated Learning epoch = 34
Test Loss=2.0076, Test accuracy=0.5125
Train:
Train Loss=1.18378356, Train acc=0.58222222
Global Federated Learning epoch = 35
Test Loss=1.9409, Test accuracy=0.5559
Train:
Train Loss=1.32103779, Train acc=0.62222222
Global Federated Learning epoch = 36
Test Loss=1.9720, Test accuracy=0.5544
Train:
Train Loss=1.28218621, Train acc=0.56000000
Global Federated Learning epoch = 37
Test Loss=1.9678, Test accuracy=0.5446
Train:
Train Loss=1.23533349, Train acc=0.63555556
Global Federated Learning epoch = 38
Test Loss=1.9498, Test accuracy=0.5545
Train:
Train Loss=1.43335028, Train acc=0.50000000
Global Federated Learning epoch = 39
Test Loss=1.9291, Test accuracy=0.5650
Train:
Train Loss=1.19080628, Train acc=0.60000000
Global Federated Learning epoch = 40
Test Loss=1.9666, Test accuracy=0.5601
Train:
Train Loss=1.27655300, Train acc=0.57777778
Global Federated Learning epoch = 41
Test Loss=1.9855, Test accuracy=0.5411
Train:
Train Loss=1.07713489, Train acc=0.60888889
Global Federated Learning epoch = 42
Test Loss=1.9495, Test accuracy=0.5652
Train:
Train Loss=1.35404345, Train acc=0.53777778
Global Federated Learning epoch = 43
Test Loss=1.9392, Test accuracy=0.5700
Train:
Train Loss=1.02540582, Train acc=0.63777778
Global Federated Learning epoch = 44
Test Loss=1.9218, Test accuracy=0.5692
Train:
Train Loss=1.18842755, Train acc=0.59333333
Global Federated Learning epoch = 45
Test Loss=1.9236, Test accuracy=0.5699
Train:
Train Loss=1.09694052, Train acc=0.60888889
Global Federated Learning epoch = 46
Test Loss=1.9122, Test accuracy=0.5812
Train:
Train Loss=1.32122970, Train acc=0.57333333
Global Federated Learning epoch = 47
Test Loss=1.9161, Test accuracy=0.5655
Train:
Train Loss=1.26719852, Train acc=0.62888889
Global Federated Learning epoch = 48
Test Loss=1.9565, Test accuracy=0.5808
Train:
Train Loss=1.18325012, Train acc=0.60888889
Global Federated Learning epoch = 49
Test Loss=1.8995, Test accuracy=0.5937
Train:
Train Loss=1.34134215, Train acc=0.55777778
Global Federated Learning epoch = 50
Test Loss=1.8756, Test accuracy=0.5955
Train:
Train Loss=0.86762036, Train acc=0.69555556
Global Federated Learning epoch = 51
Test Loss=1.9195, Test accuracy=0.5899
Train:
Train Loss=0.97150674, Train acc=0.63777778
Global Federated Learning epoch = 52
Test Loss=1.9073, Test accuracy=0.5919
Train:
Train Loss=1.33012088, Train acc=0.52666667
Global Federated Learning epoch = 53
Test Loss=1.8921, Test accuracy=0.5923
Train:
Train Loss=0.94159531, Train acc=0.66444444
Global Federated Learning epoch = 54
Test Loss=1.8506, Test accuracy=0.6039
Train:
Train Loss=0.98827976, Train acc=0.62000000
Global Federated Learning epoch = 55
Test Loss=1.8947, Test accuracy=0.6017
Train:
Train Loss=1.06258181, Train acc=0.64888889
Global Federated Learning epoch = 56
Test Loss=1.9331, Test accuracy=0.5868
Train:
Train Loss=1.21944513, Train acc=0.58888889
Global Federated Learning epoch = 57
Test Loss=1.9420, Test accuracy=0.5948
Train:
Train Loss=1.03056897, Train acc=0.59333333
Global Federated Learning epoch = 58
Test Loss=1.8507, Test accuracy=0.6134
Train:
Train Loss=1.15750675, Train acc=0.66222222
Global Federated Learning epoch = 59
Test Loss=1.8957, Test accuracy=0.6096
Train:
Train Loss=0.88404876, Train acc=0.66666667
Global Federated Learning epoch = 60
Test Loss=1.8495, Test accuracy=0.6120
Train:
Train Loss=1.12079271, Train acc=0.63333333
Global Federated Learning epoch = 61
Test Loss=1.8501, Test accuracy=0.6195
Train:
Train Loss=1.28610986, Train acc=0.60666667
Global Federated Learning epoch = 62
Test Loss=1.8856, Test accuracy=0.6171
Train:
Train Loss=0.95760518, Train acc=0.67777778
Global Federated Learning epoch = 63
Test Loss=1.8616, Test accuracy=0.6237
Train:
Train Loss=0.89708273, Train acc=0.67555556
Global Federated Learning epoch = 64
Test Loss=1.8372, Test accuracy=0.6266
Train:
Train Loss=1.13411298, Train acc=0.65555556
Global Federated Learning epoch = 65
Test Loss=1.8736, Test accuracy=0.6205
Train:
Train Loss=0.97505252, Train acc=0.66888889
Global Federated Learning epoch = 66
Test Loss=1.8592, Test accuracy=0.6209
Train:
Train Loss=1.05141917, Train acc=0.67777778
Global Federated Learning epoch = 67
Test Loss=1.8724, Test accuracy=0.6006
Train:
Train Loss=1.26682109, Train acc=0.61777778
Global Federated Learning epoch = 68
Test Loss=1.8606, Test accuracy=0.6321
Train:
Train Loss=0.85337785, Train acc=0.70222222
Global Federated Learning epoch = 69
Test Loss=1.8831, Test accuracy=0.6257
Train:
Train Loss=0.91801160, Train acc=0.70222222
Global Federated Learning epoch = 70
Test Loss=1.8719, Test accuracy=0.6339
Train:
Train Loss=1.23765076, Train acc=0.63111111
Global Federated Learning epoch = 71
Test Loss=1.8409, Test accuracy=0.6366
Train:
Train Loss=0.80518265, Train acc=0.70888889
Global Federated Learning epoch = 72
Test Loss=1.8241, Test accuracy=0.6406
Train:
Train Loss=0.89761072, Train acc=0.69111111
Global Federated Learning epoch = 73
Test Loss=1.8450, Test accuracy=0.6314
Train:
Train Loss=0.91847711, Train acc=0.69555556
Global Federated Learning epoch = 74
Test Loss=1.8487, Test accuracy=0.6368
Train:
Train Loss=1.30171385, Train acc=0.53555556
Global Federated Learning epoch = 75
Test Loss=1.8456, Test accuracy=0.6325
Train:
Train Loss=1.12795542, Train acc=0.61333333
Global Federated Learning epoch = 76
Test Loss=1.8273, Test accuracy=0.6423
Train:
Train Loss=0.84487329, Train acc=0.68444444
Global Federated Learning epoch = 77
Test Loss=1.8718, Test accuracy=0.6437
Train:
Train Loss=0.86215820, Train acc=0.69777778
Global Federated Learning epoch = 78
Test Loss=1.8418, Test accuracy=0.6396
Train:
Train Loss=1.10758585, Train acc=0.65333333
Global Federated Learning epoch = 79
Test Loss=1.8545, Test accuracy=0.6431
Train:
Train Loss=0.85918059, Train acc=0.66444444
Global Federated Learning epoch = 80
Test Loss=1.8928, Test accuracy=0.6443
Train:
Train Loss=0.92962770, Train acc=0.70666667
Global Federated Learning epoch = 81
Test Loss=1.8367, Test accuracy=0.6463
Train:
Train Loss=1.03313009, Train acc=0.73111111
Global Federated Learning epoch = 82
Test Loss=1.8492, Test accuracy=0.6446
Train:
Train Loss=1.18110310, Train acc=0.70444444
Global Federated Learning epoch = 83
Test Loss=1.8373, Test accuracy=0.6537
Train:
Train Loss=0.94358862, Train acc=0.72666667
Global Federated Learning epoch = 84
Test Loss=1.7902, Test accuracy=0.6566
Train:
Train Loss=0.85753264, Train acc=0.68444444
Global Federated Learning epoch = 85
Test Loss=1.8407, Test accuracy=0.6558
Train:
Train Loss=0.87782376, Train acc=0.72444444
Global Federated Learning epoch = 86
Test Loss=1.8270, Test accuracy=0.6602
Train:
Train Loss=1.01775344, Train acc=0.65777778
Global Federated Learning epoch = 87
Test Loss=1.8455, Test accuracy=0.6607
Train:
Train Loss=0.95975373, Train acc=0.66222222
Global Federated Learning epoch = 88
Test Loss=1.8391, Test accuracy=0.6572
Train:
Train Loss=0.73562378, Train acc=0.70888889
Global Federated Learning epoch = 89
Test Loss=1.8542, Test accuracy=0.6592
Train:
Train Loss=0.79200310, Train acc=0.74444444
Global Federated Learning epoch = 90
Test Loss=1.8375, Test accuracy=0.6647
Train:
Train Loss=0.94459957, Train acc=0.63111111
Global Federated Learning epoch = 91
Test Loss=1.8040, Test accuracy=0.6629
Train:
Train Loss=0.96883626, Train acc=0.67333333
Global Federated Learning epoch = 92
Test Loss=1.8439, Test accuracy=0.6657
Train:
Train Loss=1.08942171, Train acc=0.71111111
Global Federated Learning epoch = 93
Test Loss=1.8595, Test accuracy=0.6711
Train:
Train Loss=0.73522594, Train acc=0.72000000
Global Federated Learning epoch = 94
Test Loss=1.7953, Test accuracy=0.6748
Train:
Train Loss=1.01229978, Train acc=0.68888889
Global Federated Learning epoch = 95
Test Loss=1.7942, Test accuracy=0.6711
Train:
Train Loss=0.86881783, Train acc=0.71555556
Global Federated Learning epoch = 96
Test Loss=1.8098, Test accuracy=0.6781
Train:
Train Loss=0.98899979, Train acc=0.71555556
Global Federated Learning epoch = 97
Test Loss=1.7970, Test accuracy=0.6721
Train:
Train Loss=0.87823240, Train acc=0.66222222
Global Federated Learning epoch = 98
Test Loss=1.8246, Test accuracy=0.6786
Train:
Train Loss=1.00231885, Train acc=0.64000000
Global Federated Learning epoch = 99
Test Loss=1.8197, Test accuracy=0.6686
Train:
Train Loss=0.88826693, Train acc=0.73111111
Global Federated Learning epoch = 100
Test Loss=1.8213, Test accuracy=0.6791
Train:
Train Loss=0.70393790, Train acc=0.78666667
Global Federated Learning epoch = 101
Test Loss=1.8168, Test accuracy=0.6747
Train:
Train Loss=0.74646267, Train acc=0.74666667
Global Federated Learning epoch = 102
Test Loss=1.8157, Test accuracy=0.6829
Train:
Train Loss=0.94283428, Train acc=0.68000000
Global Federated Learning epoch = 103
Test Loss=1.8202, Test accuracy=0.6803
Train:
Train Loss=0.93741558, Train acc=0.74888889
Global Federated Learning epoch = 104
Test Loss=1.8258, Test accuracy=0.6729
Train:
Train Loss=0.77995405, Train acc=0.76666667
Global Federated Learning epoch = 105
Test Loss=1.7658, Test accuracy=0.6838
Train:
Train Loss=0.71854699, Train acc=0.78444444
Global Federated Learning epoch = 106
Test Loss=1.8154, Test accuracy=0.6721
Train:
Train Loss=0.67575409, Train acc=0.77777778
Global Federated Learning epoch = 107
Test Loss=1.8631, Test accuracy=0.6774
Train:
Train Loss=0.89136845, Train acc=0.72222222
Global Federated Learning epoch = 108
Test Loss=1.7574, Test accuracy=0.6908
Train:
Train Loss=0.72951807, Train acc=0.79777778
Global Federated Learning epoch = 109
Test Loss=1.7796, Test accuracy=0.6876
Train:
Train Loss=0.89041780, Train acc=0.69777778
Global Federated Learning epoch = 110
Test Loss=1.7679, Test accuracy=0.6899
Train:
Train Loss=0.94956928, Train acc=0.74222222
Global Federated Learning epoch = 111
Test Loss=1.8275, Test accuracy=0.6816
Train:
Train Loss=0.76064218, Train acc=0.76000000
Global Federated Learning epoch = 112
Test Loss=1.8033, Test accuracy=0.6810
Train:
Train Loss=0.82063771, Train acc=0.70222222
Global Federated Learning epoch = 113
Test Loss=1.8019, Test accuracy=0.6898
Train:
Train Loss=0.76982326, Train acc=0.72222222
Global Federated Learning epoch = 114
Test Loss=1.8065, Test accuracy=0.6918
Train:
Train Loss=0.93410004, Train acc=0.70000000
Global Federated Learning epoch = 115
Test Loss=1.8314, Test accuracy=0.6884
Train:
Train Loss=0.81288613, Train acc=0.70888889
Global Federated Learning epoch = 116
Test Loss=1.7500, Test accuracy=0.6967
Train:
Train Loss=0.83932569, Train acc=0.76222222
Global Federated Learning epoch = 117
Test Loss=1.7803, Test accuracy=0.6924
Train:
Train Loss=0.80127927, Train acc=0.75111111
Global Federated Learning epoch = 118
Test Loss=1.7630, Test accuracy=0.6942
Train:
Train Loss=1.00323156, Train acc=0.71555556
Global Federated Learning epoch = 119
Test Loss=1.7638, Test accuracy=0.6980
Train:
Train Loss=0.59792085, Train acc=0.80000000
Global Federated Learning epoch = 120
Test Loss=1.8096, Test accuracy=0.6961
Train:
Train Loss=0.98730621, Train acc=0.78000000
Global Federated Learning epoch = 121
Test Loss=1.7916, Test accuracy=0.7009
Train:
Train Loss=0.68191037, Train acc=0.76444444
Global Federated Learning epoch = 122
Test Loss=1.7534, Test accuracy=0.6961
Train:
Train Loss=1.33697686, Train acc=0.60666667
Global Federated Learning epoch = 123
Test Loss=1.8035, Test accuracy=0.6940
Train:
Train Loss=0.76247738, Train acc=0.80222222
Global Federated Learning epoch = 124
Test Loss=1.8208, Test accuracy=0.7050
Train:
Train Loss=0.84130392, Train acc=0.67555556
Global Federated Learning epoch = 125
Test Loss=1.8122, Test accuracy=0.7006
Train:
Train Loss=0.77136598, Train acc=0.74222222
Global Federated Learning epoch = 126
Test Loss=1.7609, Test accuracy=0.7061
Train:
Train Loss=0.72473516, Train acc=0.74666667
Global Federated Learning epoch = 127
Test Loss=1.7653, Test accuracy=0.6997
Train:
Train Loss=0.68493576, Train acc=0.82888889
Global Federated Learning epoch = 128
Test Loss=1.7870, Test accuracy=0.6984
Train:
Train Loss=0.82242663, Train acc=0.70888889
Global Federated Learning epoch = 129
Test Loss=1.7675, Test accuracy=0.7077
Train:
Train Loss=0.55332763, Train acc=0.75777778
Global Federated Learning epoch = 130
Test Loss=1.8108, Test accuracy=0.7072
Train:
Train Loss=1.05763827, Train acc=0.72444444
Global Federated Learning epoch = 131
Test Loss=1.7641, Test accuracy=0.7089
Train:
Train Loss=0.64746760, Train acc=0.79555556
Global Federated Learning epoch = 132
Test Loss=1.7549, Test accuracy=0.7045
Train:
Train Loss=0.65450943, Train acc=0.80222222
Global Federated Learning epoch = 133
Test Loss=1.7830, Test accuracy=0.7080
Train:
Train Loss=0.85267922, Train acc=0.74888889
Global Federated Learning epoch = 134
Test Loss=1.7870, Test accuracy=0.7034
Train:
Train Loss=0.73928875, Train acc=0.75333333
Global Federated Learning epoch = 135
Test Loss=1.7553, Test accuracy=0.7102
Train:
Train Loss=0.61265180, Train acc=0.78222222
Global Federated Learning epoch = 136
Test Loss=1.7523, Test accuracy=0.7124
Train:
Train Loss=0.44610789, Train acc=0.84666667
Global Federated Learning epoch = 137
Test Loss=1.7847, Test accuracy=0.7071
Train:
Train Loss=0.72062421, Train acc=0.77111111
Global Federated Learning epoch = 138
Test Loss=1.7577, Test accuracy=0.7130
Train:
Train Loss=1.17030712, Train acc=0.66000000
Global Federated Learning epoch = 139
Test Loss=1.7843, Test accuracy=0.7128
Train:
Train Loss=0.53333345, Train acc=0.80444444
Global Federated Learning epoch = 140
Test Loss=1.8477, Test accuracy=0.7046
Train:
Train Loss=0.52741401, Train acc=0.79555556
Global Federated Learning epoch = 141
Test Loss=1.7680, Test accuracy=0.7129
Train:
Train Loss=0.70315865, Train acc=0.74444444
Global Federated Learning epoch = 142
Test Loss=1.7988, Test accuracy=0.7107
Train:
Train Loss=0.49444137, Train acc=0.84666667
Global Federated Learning epoch = 143
Test Loss=1.8023, Test accuracy=0.7071
Train:
Train Loss=0.50834230, Train acc=0.82444444
Global Federated Learning epoch = 144
Test Loss=1.7613, Test accuracy=0.7153
Train:
Train Loss=0.62030879, Train acc=0.79555556
Global Federated Learning epoch = 145
Test Loss=1.7660, Test accuracy=0.7132
Train:
Train Loss=0.59195822, Train acc=0.77333333
Global Federated Learning epoch = 146
Test Loss=1.7968, Test accuracy=0.7116
Train:
Train Loss=0.88626042, Train acc=0.76000000
Global Federated Learning epoch = 147
Test Loss=1.7855, Test accuracy=0.7189
Train:
Train Loss=1.08963898, Train acc=0.74444444
Global Federated Learning epoch = 148
Test Loss=1.7538, Test accuracy=0.7179
Train:
Train Loss=0.76195917, Train acc=0.74666667
Global Federated Learning epoch = 149
Test Loss=1.7527, Test accuracy=0.7178
Train:
Train Loss=0.75736266, Train acc=0.72666667
Global Federated Learning epoch = 150
Test Loss=1.7628, Test accuracy=0.7138
Train:
Train Loss=0.72319508, Train acc=0.75111111
Global Federated Learning epoch = 151
Test Loss=1.8088, Test accuracy=0.7099
Train:
Train Loss=0.78828249, Train acc=0.75333333
Global Federated Learning epoch = 152
Test Loss=1.8151, Test accuracy=0.7147
Train:
Train Loss=0.82053678, Train acc=0.80222222
Global Federated Learning epoch = 153
Test Loss=1.7415, Test accuracy=0.7176
Train:
Train Loss=0.53223664, Train acc=0.85333333
Global Federated Learning epoch = 154
Test Loss=1.7932, Test accuracy=0.7182
Train:
Train Loss=0.63067030, Train acc=0.82666667
Global Federated Learning epoch = 155
Test Loss=1.7632, Test accuracy=0.7221
Train:
Train Loss=0.92258913, Train acc=0.61333333
Global Federated Learning epoch = 156
Test Loss=1.7778, Test accuracy=0.7190
Train:
Train Loss=0.60861639, Train acc=0.77777778
Global Federated Learning epoch = 157
Test Loss=1.7838, Test accuracy=0.7225
Train:
Train Loss=0.70940235, Train acc=0.78000000
Global Federated Learning epoch = 158
Test Loss=1.7761, Test accuracy=0.7205
Train:
Train Loss=0.60261337, Train acc=0.79777778
Global Federated Learning epoch = 159
Test Loss=1.7471, Test accuracy=0.7211
Train:
Train Loss=0.58822722, Train acc=0.80888889
Global Federated Learning epoch = 160
Test Loss=1.7832, Test accuracy=0.7240
Train:
Train Loss=0.76010894, Train acc=0.74666667
Global Federated Learning epoch = 161
Test Loss=1.8010, Test accuracy=0.7252
Train:
Train Loss=0.61039454, Train acc=0.77111111
Global Federated Learning epoch = 162
Test Loss=1.7585, Test accuracy=0.7192
Train:
Train Loss=0.53269583, Train acc=0.81111111
Global Federated Learning epoch = 163
Test Loss=1.7355, Test accuracy=0.7238
Train:
Train Loss=0.57217206, Train acc=0.81555556
Global Federated Learning epoch = 164
Test Loss=1.7288, Test accuracy=0.7267
Train:
Train Loss=0.67719882, Train acc=0.82888889
Global Federated Learning epoch = 165
Test Loss=1.7274, Test accuracy=0.7252
Train:
Train Loss=1.04168978, Train acc=0.72888889
Global Federated Learning epoch = 166
Test Loss=1.7461, Test accuracy=0.7241
Train:
Train Loss=0.83332676, Train acc=0.71555556
Global Federated Learning epoch = 167
Test Loss=1.7621, Test accuracy=0.7215
Train:
Train Loss=0.63440858, Train acc=0.78888889
Global Federated Learning epoch = 168
Test Loss=1.7476, Test accuracy=0.7262
Train:
Train Loss=0.48241605, Train acc=0.85777778
Global Federated Learning epoch = 169
Test Loss=1.7511, Test accuracy=0.7239
Train:
Train Loss=0.43771722, Train acc=0.85777778
Global Federated Learning epoch = 170
Test Loss=1.8220, Test accuracy=0.7187
Train:
Train Loss=0.48824373, Train acc=0.82000000
Global Federated Learning epoch = 171
Test Loss=1.7472, Test accuracy=0.7275
Train:
Train Loss=0.57629332, Train acc=0.76888889
Global Federated Learning epoch = 172
Test Loss=1.7835, Test accuracy=0.7240
Train:
Train Loss=0.55681159, Train acc=0.78222222
Global Federated Learning epoch = 173
Test Loss=1.7648, Test accuracy=0.7270
Train:
Train Loss=0.70599432, Train acc=0.78888889
Global Federated Learning epoch = 174
Test Loss=1.7067, Test accuracy=0.7298
Train:
Train Loss=0.75781517, Train acc=0.83333333
Global Federated Learning epoch = 175
Test Loss=1.7806, Test accuracy=0.7257
Train:
Train Loss=0.85842014, Train acc=0.75555556
Global Federated Learning epoch = 176
Test Loss=1.7969, Test accuracy=0.7199
Train:
Train Loss=0.58013633, Train acc=0.80444444
Global Federated Learning epoch = 177
Test Loss=1.7512, Test accuracy=0.7230
Train:
Train Loss=0.70675075, Train acc=0.77555556
Global Federated Learning epoch = 178
Test Loss=1.7748, Test accuracy=0.7331
Train:
Train Loss=0.54942298, Train acc=0.84888889
Global Federated Learning epoch = 179
Test Loss=1.7707, Test accuracy=0.7286
Train:
Train Loss=0.57932143, Train acc=0.78000000
Global Federated Learning epoch = 180
Test Loss=1.7898, Test accuracy=0.7270
Train:
Train Loss=0.62044566, Train acc=0.78222222
Global Federated Learning epoch = 181
Test Loss=1.7482, Test accuracy=0.7356
Train:
Train Loss=0.66170069, Train acc=0.81111111
Global Federated Learning epoch = 182
Test Loss=1.7665, Test accuracy=0.7322
Train:
Train Loss=0.51486816, Train acc=0.82444444
Global Federated Learning epoch = 183
Test Loss=1.7381, Test accuracy=0.7345
Train:
Train Loss=0.83458042, Train acc=0.75333333
Global Federated Learning epoch = 184
Test Loss=1.7935, Test accuracy=0.7333
Train:
Train Loss=1.16214754, Train acc=0.60888889
Global Federated Learning epoch = 185
Test Loss=1.7732, Test accuracy=0.7260
Train:
Train Loss=0.42825856, Train acc=0.86888889
Global Federated Learning epoch = 186
Test Loss=1.7760, Test accuracy=0.7334
Train:
Train Loss=0.53698139, Train acc=0.83555556
Global Federated Learning epoch = 187
Test Loss=1.6858, Test accuracy=0.7344
Train:
Train Loss=0.43600881, Train acc=0.86222222
Global Federated Learning epoch = 188
Test Loss=1.6885, Test accuracy=0.7358
Train:
Train Loss=0.50015684, Train acc=0.80222222
Global Federated Learning epoch = 189
Test Loss=1.7294, Test accuracy=0.7331
Train:
Train Loss=0.98999114, Train acc=0.66000000
Global Federated Learning epoch = 190
Test Loss=1.7341, Test accuracy=0.7334
Train:
Train Loss=0.47850785, Train acc=0.83555556
Global Federated Learning epoch = 191
Test Loss=1.7154, Test accuracy=0.7296
Train:
Train Loss=0.64905402, Train acc=0.81777778
Global Federated Learning epoch = 192
Test Loss=1.7560, Test accuracy=0.7338
Train:
Train Loss=0.50762876, Train acc=0.84222222
Global Federated Learning epoch = 193
Test Loss=1.7406, Test accuracy=0.7344
Train:
Train Loss=0.80234854, Train acc=0.74444444
Global Federated Learning epoch = 194
Test Loss=1.7373, Test accuracy=0.7357
Train:
Train Loss=0.84476685, Train acc=0.74222222
Global Federated Learning epoch = 195
Test Loss=1.7497, Test accuracy=0.7314
Train:
Train Loss=0.46995007, Train acc=0.84888889
Global Federated Learning epoch = 196
Test Loss=1.7340, Test accuracy=0.7333
Train:
Train Loss=0.78372301, Train acc=0.73777778
Global Federated Learning epoch = 197
Test Loss=1.7332, Test accuracy=0.7325
Train:
Train Loss=0.76067319, Train acc=0.81333333
Global Federated Learning epoch = 198
Test Loss=1.7768, Test accuracy=0.7244
Train:
Train Loss=0.57009632, Train acc=0.79555556
Global Federated Learning epoch = 199
Test Loss=1.7754, Test accuracy=0.7325
Train:
Train Loss=1.06652165, Train acc=0.64000000
Global Federated Learning epoch = 200
Test Loss=1.6888, Test accuracy=0.7388
Train:
Train Loss=0.82611478, Train acc=0.74222222
Global Federated Learning epoch = 201
Test Loss=1.7514, Test accuracy=0.7317
Train:
Train Loss=0.55414256, Train acc=0.85555556
Global Federated Learning epoch = 202
Test Loss=1.7552, Test accuracy=0.7384
Train:
Train Loss=1.05581183, Train acc=0.79333333
Global Federated Learning epoch = 203
Test Loss=1.7025, Test accuracy=0.7334
Train:
Train Loss=0.77727049, Train acc=0.80444444
Global Federated Learning epoch = 204
Test Loss=1.7830, Test accuracy=0.7333
Train:
Train Loss=0.83833912, Train acc=0.79777778
Global Federated Learning epoch = 205
Test Loss=1.7118, Test accuracy=0.7382
Train:
Train Loss=0.34734259, Train acc=0.88222222
Global Federated Learning epoch = 206
Test Loss=1.7436, Test accuracy=0.7371
Train:
Train Loss=0.67305630, Train acc=0.81555556
Global Federated Learning epoch = 207
Test Loss=1.6834, Test accuracy=0.7378
Train:
Train Loss=0.41258239, Train acc=0.87111111
Global Federated Learning epoch = 208
Test Loss=1.7520, Test accuracy=0.7346
Train:
Train Loss=0.70744257, Train acc=0.77555556
Global Federated Learning epoch = 209
Test Loss=1.7631, Test accuracy=0.7386
Train:
Train Loss=0.41745306, Train acc=0.82444444
Global Federated Learning epoch = 210
Test Loss=1.7217, Test accuracy=0.7391
Train:
Train Loss=0.65759061, Train acc=0.75777778
Global Federated Learning epoch = 211
Test Loss=1.7322, Test accuracy=0.7352
Train:
Train Loss=0.65151373, Train acc=0.84222222
Global Federated Learning epoch = 212
Test Loss=1.7611, Test accuracy=0.7376
Train:
Train Loss=0.79347867, Train acc=0.80000000
Global Federated Learning epoch = 213
Test Loss=1.7716, Test accuracy=0.7386
Train:
Train Loss=0.45171878, Train acc=0.87111111
Global Federated Learning epoch = 214
Test Loss=1.7744, Test accuracy=0.7355
Train:
Train Loss=0.56323759, Train acc=0.79777778
Global Federated Learning epoch = 215
Test Loss=1.7046, Test accuracy=0.7382
Train:
Train Loss=0.61783962, Train acc=0.84888889
Global Federated Learning epoch = 216
Test Loss=1.7173, Test accuracy=0.7407
Train:
Train Loss=0.58295745, Train acc=0.84000000
Global Federated Learning epoch = 217
Test Loss=1.7529, Test accuracy=0.7346
Train:
Train Loss=0.55267621, Train acc=0.84222222
Global Federated Learning epoch = 218
Test Loss=1.6718, Test accuracy=0.7428
Train:
Train Loss=0.43115703, Train acc=0.84888889
Global Federated Learning epoch = 219
Test Loss=1.7010, Test accuracy=0.7436
Train:
Train Loss=1.09452225, Train acc=0.67111111
Global Federated Learning epoch = 220
Test Loss=1.7077, Test accuracy=0.7388
Train:
Train Loss=0.66703136, Train acc=0.76000000
Global Federated Learning epoch = 221
Test Loss=1.7377, Test accuracy=0.7420
Train:
Train Loss=0.57655139, Train acc=0.80444444
Global Federated Learning epoch = 222
Test Loss=1.6971, Test accuracy=0.7415
Train:
Train Loss=0.64397921, Train acc=0.86444444
Global Federated Learning epoch = 223
Test Loss=1.7553, Test accuracy=0.7414
Train:
Train Loss=0.38909996, Train acc=0.86222222
Global Federated Learning epoch = 224
Test Loss=1.7302, Test accuracy=0.7432
Train:
Train Loss=0.70279008, Train acc=0.73555556
Global Federated Learning epoch = 225
Test Loss=1.7198, Test accuracy=0.7421
Train:
Train Loss=0.73794552, Train acc=0.82222222
Global Federated Learning epoch = 226
Test Loss=1.7051, Test accuracy=0.7409
Train:
Train Loss=0.47702065, Train acc=0.81555556
Global Federated Learning epoch = 227
Test Loss=1.7334, Test accuracy=0.7460
Train:
Train Loss=0.39780469, Train acc=0.90666667
Global Federated Learning epoch = 228
Test Loss=1.7506, Test accuracy=0.7435
Train:
Train Loss=0.76913827, Train acc=0.73555556
Global Federated Learning epoch = 229
Test Loss=1.7141, Test accuracy=0.7416
Train:
Train Loss=0.68435195, Train acc=0.78000000
Global Federated Learning epoch = 230
Test Loss=1.7076, Test accuracy=0.7463
Train:
Train Loss=0.67696523, Train acc=0.85777778
Global Federated Learning epoch = 231
Test Loss=1.7470, Test accuracy=0.7403
Train:
Train Loss=0.43146603, Train acc=0.87111111
Global Federated Learning epoch = 232
Test Loss=1.7322, Test accuracy=0.7443
Train:
Train Loss=0.59395871, Train acc=0.83333333
Global Federated Learning epoch = 233
Test Loss=1.7148, Test accuracy=0.7426
Train:
Train Loss=0.60797914, Train acc=0.87333333
Global Federated Learning epoch = 234
Test Loss=1.7208, Test accuracy=0.7469
Train:
Train Loss=0.60988831, Train acc=0.81777778
Global Federated Learning epoch = 235
Test Loss=1.6688, Test accuracy=0.7436
Train:
Train Loss=0.56054892, Train acc=0.81111111
Global Federated Learning epoch = 236
Test Loss=1.7377, Test accuracy=0.7434
Train:
Train Loss=0.50503094, Train acc=0.86666667
Global Federated Learning epoch = 237
Test Loss=1.6446, Test accuracy=0.7450
Train:
Train Loss=0.62336044, Train acc=0.80666667
Global Federated Learning epoch = 238
Test Loss=1.7245, Test accuracy=0.7450
Train:
Train Loss=0.35658028, Train acc=0.87777778
Global Federated Learning epoch = 239
Test Loss=1.7200, Test accuracy=0.7489
Train:
Train Loss=0.43100600, Train acc=0.83555556
Global Federated Learning epoch = 240
Test Loss=1.6676, Test accuracy=0.7455
Train:
Train Loss=0.96691099, Train acc=0.78222222
Global Federated Learning epoch = 241
Test Loss=1.7251, Test accuracy=0.7407
Train:
Train Loss=0.67587921, Train acc=0.82444444
Global Federated Learning epoch = 242
Test Loss=1.7189, Test accuracy=0.7474
Train:
Train Loss=0.39906301, Train acc=0.86222222
Global Federated Learning epoch = 243
Test Loss=1.6687, Test accuracy=0.7459
Train:
Train Loss=0.53927217, Train acc=0.90222222
Global Federated Learning epoch = 244
Test Loss=1.7138, Test accuracy=0.7473
Train:
Train Loss=0.66252906, Train acc=0.87333333
Global Federated Learning epoch = 245
Test Loss=1.7286, Test accuracy=0.7469
Train:
Train Loss=0.68520103, Train acc=0.75777778
Global Federated Learning epoch = 246
Test Loss=1.7277, Test accuracy=0.7487
Train:
Train Loss=0.87283554, Train acc=0.78666667
Global Federated Learning epoch = 247
Test Loss=1.7497, Test accuracy=0.7477
Train:
Train Loss=0.51451351, Train acc=0.83111111
Global Federated Learning epoch = 248
Test Loss=1.7303, Test accuracy=0.7513
save path H:\python\FL-class-discriminative-pruning\ckpt\retrained\resnet44\seed_acc75.13_epoch248_2024-03-06 03-06-02.pth"""

epoch_acc = []
list = str.split('epoch = ')
# print(list)
for i in range(1, len(list)):
    e = list[i].split('\n')
    epoch = int(e[0])
    el = e[1].split('=')
    acc = float(el[-1])
    epoch_acc.append((epoch, acc))
print(epoch_acc)
epochs, accuracies = zip(*epoch_acc)

# 将数据存储到Excel表中
data = {'Epoch': epochs, 'Accuracy': accuracies}
df = pd.DataFrame(data)
excel_file = 'cifar10_resnet44_retrained.xlsx'
df.to_excel('./excel/' + excel_file, index=False)

